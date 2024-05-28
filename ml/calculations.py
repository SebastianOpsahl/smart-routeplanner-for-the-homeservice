import numpy as np
import torch
import torch.nn.functional as F
from structures.structures import Solution, EnvRoute, EnvVisit, Route, Visit, RouteOut, SolutionOut, VisitOut
from datetime import timedelta, datetime

def action(pointer_network, visit_features):
    if visit_features.dim() == 1:
        visit_features = visit_features.unsqueeze(0)

    visit1_index, visit2_index, before_after, log_prob1, log_prob2, log_prob_ba = pointer_network(visit_features)

    return visit1_index, visit2_index, before_after, log_prob1, log_prob2, log_prob_ba

def compute_gae(next_value, rewards, masks, values, gamma=0.98, tau=0.95):
    '''Computes the Generalized Advantage Estimation (GAE) which quantify how much 
    better or worse actions turned out compared to a baseline. '''
    gae = 0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
        next_value = values[step]
    return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

def update_policy(pointer_network, critic, optimizer_pointer_network, optimizer_critic, rollout_buffer, 
                epochs=4, clip_param=0.1):
    '''Updates the policy based on how well the new policy performs. The change of the update is based on a
    clipping parameter'''
    
    states = torch.stack(rollout_buffer.states)
    logprobs = torch.stack(rollout_buffer.logprobs)
    rewards = torch.stack(rollout_buffer.rewards)
    is_terminals = torch.stack(rollout_buffer.is_terminals)

    states = states.squeeze(1)
    values = critic(states).detach()

    # Calculate the value for the next state for each trajectory
    next_values = critic(states[-1].unsqueeze(0)).detach().squeeze() * (1 - is_terminals[-1])

    # Computes returns and advantages
    returns, advantages = compute_gae(next_values, rewards, is_terminals, values)

    # Normalizes advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ppo_update(pointer_network, critic, optimizer_pointer_network, optimizer_critic, states, logprobs,
            returns, advantages, epochs, clip_param)

    # Reset the rollout buffer for the next collection phase
    rollout_buffer.clear()

def ppo_update(pointer_network, critic, optimizer_pointer_network, optimizer_critic, states, 
            logprobs, returns, advantages, epochs, clip_param):
    for _ in range(epochs):
        '''Function to compute and update the networks. Based on performance under previous previous policies
        contra the current one'''

        # Computes the return under the current policy
        logprobs = []
        for state in states.squeeze(1):
            _, _, _, log_prob1, log_prob2, log_prob_ba = action(pointer_network, state.unsqueeze(0))
            logprobs.append(torch.stack([log_prob1, log_prob2, log_prob_ba.flatten()]))
        logprobs = torch.stack(logprobs).mean(dim=1)

        # Calculates the ratio of new to old probabilities
        ratios = torch.exp(logprobs - logprobs)  
       
        # Calculates surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Calculates the critic's value loss
        values = critic(states).squeeze()
        value_loss = F.mse_loss(values, returns)

        # Performs gradient descent steps for both networks
        optimizer_pointer_network.zero_grad()
        policy_loss.backward()
        optimizer_pointer_network.step()

        optimizer_critic.zero_grad()
        value_loss.backward()
        optimizer_critic.step()

def normalize(solution: Solution):
    '''Normalizes value to a scale from 0-1 to make understandable for neural networks and learning algorithms'''
    scaled_matrix = scale_matrix(solution.matrix)
    scaled_routes = []

    # X is used to default all shift starts to 00:00
    x = solution.shift_time[0] - solution.shift_time[0].replace(hour=0, minute=0, second=0, microsecond=0)
    scaled_shift_time = [datetime_to_scale(shift_time, x) for shift_time in solution.shift_time] 
    scaled_break_time = [datetime_to_scale(break_time, x) for break_time in solution.break_time] 

    for route in solution.routes:
        scaled_visits = []
        for visit in route.visits:
            double_staffed = np.float32(1.0 if visit.double_staffed else 0.0)
            start_time = datetime_to_scale(visit.start_time, x)
            end_time = datetime_to_scale(visit.end_time, x)
            task_time = timedelta_to_scale(visit.task_time)

            scaled_visit = EnvVisit(
                visit_id=visit.visit_id,
                matrix_index=visit.matrix_index,
                patient=visit.patient,
                start_time=start_time,
                end_time=end_time,
                task_time=task_time,
                double_staffed=double_staffed,
                tasks=visit.tasks
            )
            scaled_visits.append(scaled_visit)

        scaled_route = EnvRoute(visits=scaled_visits)
        scaled_routes.append(scaled_route)

    return Solution(scaled_shift_time, scaled_break_time, scaled_routes, scaled_matrix)

def scale_matrix(matrix: list[list[timedelta]]):
    '''Scales matrix from using timedelta to float'''
    return [[timedelta_to_scale(td) for td in row] for row in matrix]
        
def datetime_to_scale(dt: datetime, x: timedelta):
    '''Scales datetime values to percentage of day'''
    dt = dt - x
    seconds_since_midnight: np.float32 = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

    return round(seconds_since_midnight / 86400, 7)

def timedelta_to_scale(td: timedelta):
    '''Scales timedelta values to percentage of day'''
    return round(td.total_seconds() / 86400, 7)

def unormalize(solution: Solution, shift_time: list[datetime], break_time: list[datetime]):
    unormalized_routes = []

    x = shift_time[0]
    
    for route in solution.routes:
        unormalized_visits = []
        for visit in route.visits:
            start_time = scale_to_datetime(visit.start_time, x)
            end_time = scale_to_datetime(visit.start_time, x)
            task_time = scale_to_timedelta(visit.task_time)

            unormalized_visit = VisitOut(
                patient=visit.patient,
                start_time=start_time,
                end_time=end_time,
                task_time=task_time,
                double_staffed=bool(visit.double_staffed > 0.5),
                tasks=visit.tasks
            )
            unormalized_visits.append(unormalized_visit)

        unormalized_route = RouteOut(visits=unormalized_visits)
        unormalized_routes.append(unormalized_route)
    return SolutionOut(shift_time=shift_time, break_time=break_time, routes=unormalized_routes)

# Utility functions
def scale_to_datetime(scale, base_datetime):
    seconds_since_midnight = scale * 86400
    return base_datetime + timedelta(seconds=seconds_since_midnight)

def unnormalize_matrix(matrix):
    '''Unscales matrix from float to using timedelta'''
    return [[scale_to_timedelta(value) for value in row] for row in matrix]

def scale_to_timedelta(scale: float):
    '''Converts scaled percentage of day back to timedelta, rounded to nearest second.'''
    total_seconds = scale * 86400
    rounded_seconds = round(total_seconds)
    return timedelta(seconds=rounded_seconds)