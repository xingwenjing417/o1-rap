import io
import os
import random
import warnings
from collections import defaultdict, Counter
from copy import deepcopy
from tqdm import tqdm, trange

from .mcts import MCTS, MCTSNode
#from utils_lzq.gpt4o import concurrent_apply_gpt4o,workflow_gpt4o_gen,workflow_gpt4o_transit,workflow_gpt4o_reward
from utils_lzq.gpt4 import concurrent_apply_gpt4

def is_terminal_prompt_or_action(prompt):
    prompt = prompt.split('\n\n')[-1]
    if 'finished' in prompt.lower():
        return True
    return False

class ReasoningMCTSNode(MCTSNode):

    #@property 装饰器将方法变成属性，使得可以通过 node.visited 的方式访问 self._visited 的值，而不需要显式地调用方法。
    @property
    def visited(self):
        return self._visited

    def __init__(self, state, gen_fn, reward_fn, depth, r1, max_depth,init_state,
                 parent: 'ReasoningMCTSNode' = None, action=None):
        self._conf = None
        self.children = []
        self.state = state
        self.action = action
        self.gen_fn = gen_fn
        self.reward_fn = reward_fn
        self.depth = depth
        self.max_depth = max_depth
        self._r1 = r1
        if action == 'finish':
            self._r1 += 0.01
        self._ans_list = None
        self._visited = True # we do not need to visit again for ProntoQA MCTS settings
        self.parent = parent
        self._terminal = False
        self.init_state = init_state
    
    def _child_node(self, action, next_state, r1):
        return ReasoningMCTSNode(next_state, self.gen_fn, self.reward_fn, self.depth + 1,
                                 r1, self.max_depth,self.init_state, parent=self, action=action)
    

    def _get_children(self):
        self._visited = True
        self._calculate_reward()
        if self.is_terminal or self._r1 <= -1 or self.depth == self.max_depth:
            return self.children
        for action, next_state, reward in self.gen_fn(self.state, self.depth, self.init_state):
            self.children.append(self._child_node(action, next_state, reward))
        return self.children
    
    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    #
    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _calculate_reward(self):
        return self._r1

    @property
    def is_terminal(self):
        return self.action is not None and is_terminal_prompt_or_action(self.action)

    @property
    def reward(self):
         return self._r1

    def print(self, mcts: MCTS, file=None):
        def pprint(*args):
            if file is None:
                tqdm.write(*args)
            else:
                print(*args, file=file)
        p1 = '-' * (4 * self.depth - 4)
        prefix = ' ' * (4 * self.depth - 4)
        if self.action is None:
            action = f'S.{self.depth}: {self.state}'
        else:
            action = f'A.{self.depth}: {self.action}'
        pprint(p1 + action)
        pprint(prefix + f'R: {self.reward:.3f} ; N: {mcts.N[self]} ; M: {mcts.M[self]:.3f}')
        if self.action is not None and self.state is not None:
            term = '\u25A1' if self.is_terminal else ''
            state = f'S.{self.depth}: {self.state} ; r1: {self._r1:.3f} {term}'
            pprint(prefix + state)
        for child in self.children:
            child.print(mcts, file)
        if self.depth == 1:
            pprint("=" * 12)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.gen_fn is None or self.reward_fn is None:
            warnings.warn('MCTSNode loaded from pickle is read-only; Do not further roll out the tree!')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['gen_fn'] = None
        state['reward_fn'] = None
        return state


def reasoning_mcts_search(query: str,
                          gen_sp,
                          transit_sp,
                          reward_sp,
                          mcts_rollouts,
                          w_exp,
                          max_depth,
                          client,
                          logging=False
                          ):
    init_state = query
    if logging:
        os.path.exists('agent.log') and os.remove('agent.log')
        os.path.exists('wm.log') and os.remove('wm.log')


    next_action_state_cache = {}

    def gen_fn(state, cur_depth, init_state):
        if state in next_action_state_cache:
            return next_action_state_cache[state]
        user_input = f"Init_state:{init_state},Curr_state:{state},Depth:{cur_depth}"
        '''
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": gen_sp_input},
                {"role": "user", "content": agent_input},
            ],
            stream=False,
            timeout=100000
        )
        agent_output = response.choices[0].message.content
        '''
        agent_output = concurrent_apply_gpt4(user_input, gen_sp)
        ac_count = agent_output.count("action")
        agent_output = agent_output.strip().split('生成动作：')[-1]
        agent_output = agent_output.strip().split('action:')[-ac_count:]
        agent_output = [item for item in agent_output if item != '']
        agent_output_counter = Counter(agent_output)

        if logging:
            with open('agent.log', 'a') as f:
                
                for o in agent_output:
                    print(f'{o}', file=f)
                print('=' * 20, file=f)

        next_state_dict = defaultdict(lambda: [])
        
        for action in agent_output:
            next_state = transit_fn(state, action, init_state)
            next_state_dict[next_state].append((action, agent_output_counter[action]))
        
        ret_actions, ret_next_states = [], []
        for next_state, actions in next_state_dict.items():
            ret_actions.append(max(actions, key=lambda x: x[1])[0])
            ret_next_states.append(next_state)
        rewards = reward_fn(state, ret_actions, ret_next_states,init_state) if len(ret_actions) else []
        ret = list(zip(ret_actions, ret_next_states, rewards))
        next_action_state_cache[state] = ret
        return ret

    def transit_fn(state, action, init_state):
        user_input = f"Init_state:{init_state},Curr_state:{state},Action:{action}"
        world_output = concurrent_apply_gpt4(user_input, transit_sp)
        result = world_output.strip().split('output_prefix')[-1]

        return result

    def reward_fn(state, actions, next_states,init_state):
        world_inputs = []
        for action, next_state in zip(actions, next_states):
            agent_input = f"Init_state:{init_state},Curr_state:{state},Action:{action},Next_state:{next_state}"
            world_inputs.append(agent_input)
        world_outputs = []
        for w_input in world_inputs:
            w_output = concurrent_apply_gpt4(w_input, reward_sp)
            w_output = w_output.strip().split('result:')[-1]
            w_output = int(w_output)
            world_outputs.append(w_output)
        wo_sum = sum(world_outputs)
        rewards = [wo / wo_sum for wo in world_outputs]
        print(f"reward list is {rewards}")



        if logging:
            with open('wm_reward.log', 'a') as f:
                for world_input, reward in zip(world_inputs, rewards):
                    print(world_input.split('\n\n')[-1], file=f)
                    print(reward, file=f)
                    print('='*20, file=f)
        return rewards

    mcts = MCTS(w_exp=w_exp, prior=True, aggr_reward='mean', aggr_child='max')
    root = ReasoningMCTSNode(init_state, gen_fn, None,
                             depth=1, r1=1, max_depth=max_depth, init_state=init_state, parent=None)
    trajs = []
    outputs = []
    trees = []
    
    for i in (pbar := trange(mcts_rollouts, disable=bool(int(os.environ.get("LOCAL_RANK", -1))), position=0)):
        print(f"mcts {i}th round,start rollout")
        mcts.rollout(root)
        print(f"now stop rollout,mcts {i}th round")
        root.print(mcts)
        max_n, max_r = mcts.max_mean_terminal(root)
        cur = max_n.parent
        traj = []
        if cur is not None:
            while cur != root:
                traj.append(cur.state)
                traj.append(cur.action)
                cur = cur.parent
            traj.append(cur.state)
            traj = list(reversed(traj))  
        trajs.append(traj)
        
        for i in ['true', 'false']:
            if i in max_n.state.lower():
                temp_r = i
                break
        else:
            temp_r = 'none'
        outputs.append(temp_r)
        pbar.set_description(f'{max_r:.3f} {temp_r}')
        tree_copy = deepcopy(root)
        tree_copy.Q = dict(mcts.Q)
        tree_copy.N = dict(mcts.N)
        tree_copy.M = dict(mcts.M)
        trees.append(tree_copy)

    with io.StringIO() as f:
        root.print(mcts, file=f)
        tree = f.getvalue()
    return trajs, tree, trees, outputs