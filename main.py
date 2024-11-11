import math
import os
import re
import json
import numpy as np
import pandas as pd
import time
from retry import retry
from config import *
from dotenv import load_dotenv
from openai import OpenAI
from utils import *

if not os.path.exists("./output"):
    os.makedirs("./output")
    os.makedirs("./output/json/")

load_dotenv()

clients = []
times = time.time()

def create_client(i):
    global clients
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv('GROQ_API_KEY')
    )

    try:
        if i == 0:
            # This is the SVA Generator
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SVA_SYSTEM_PROMPT}
                ],
                temperature=0.95,
                timeout=15
            )
            clients.append(client)
        elif i == 1:
            # This is the critic
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT}
                ],
                temperature=0.95,
                timeout=15
            )
            clients.append(client)
    except:
        print("client creation failed!")
        exit()

def create_clients():
    global clients
    for i in range(2):
        create_client(i)

@retry()
def generate(prompt,history=[],timeout = 50,truncate=True,model="critic"):
    global clients

    time0 = time.time()
    history_ = [{"role": "user" if i%2 == 0 else 'assistant', "content": h} for i,h in enumerate(history)]

    if model == "critic":
        llm = clients[1]
    elif model == "generator":
        llm = clients[0]

    print

    completion = llm.chat.completions.create(
        model = MODEL_NAME,
        messages = history_ + [
            {"role": "user", "content": prompt}
        ],
        temperature=0.95,
        timeout = timeout
    )

    print(f'response received! time taken: {time.time()-time0} seconds.')
    return completion.choices[0].message.content, list(history) + [prompt,completion.choices[0].message.content]

@retry()
def cal_reward(question,ans,history):
    query = f'Question: {question}\nAnswer:{ans}\nAnalyze this Answer Strictly and Critic, point out every flaw for every possible imperfect to minus every possible score! You need to be very harsh and mean in calculating grades, and never give full marks to ensure that the marks are authoritative. \nOutput a score between [-100,+100]. \nResponse format:\n[Analysis]...[Score]...'
    ret = generate(query, history, model="critic")

    score = ret[0].split('Score')[-1]
    scores = pattern.findall(score)

    if not scores:
        raise Exception('Failed to get reward from answer')
    else:
        ret = float(scores[-1])
        if ret >= 95:
            ret = 50
        return ret

@retry()
def get_weak_answer(question,history):
    query = f'{question}\nThe response should begin with [Reasoning Process]... [Verification]... and end with [Answer]... \nLet\'s think step by step.'
    return generate(query, history, model = "generator")

def get_weak_hints(question,weak_answer,history=[]):
    query = f'{question}.\nSince we have a weak Answer from the specifications: {weak_answer}, could you provide me with a relection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score!\nLet\'s think step by step.'
    return generate(query,history, model="critic")

def get_better_answer(question,weak_answer,hint,history=[]):
    query = f'{question}.\nPlease refine your old answer: {weak_answer}.\n\nAccording to the Feedback: {hint}. The response should begin with [reasoning process]...[Verification]... and end with end with [Answer]\nLet\'s think step by step.'
    return generate(query,history, model="generator")

datas = []
pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')

def sort_answers_and_rewards(answers, rewards):
    # Zip answers and rewards together
    answer_reward_pairs = zip(answers, rewards)
    
    # Sort pairs by rewards
    sorted_pairs = sorted(answer_reward_pairs, key=lambda x: x[1], reverse=True)
    
    # Extract sorted answers and rewards
    sorted_answers = [pair[0] for pair in sorted_pairs]
    sorted_rewards = [pair[1] for pair in sorted_pairs]
    
    return sorted_answers, sorted_rewards

def filter_mature_node(childs, to_explore, to_explore_reward, max_expand=3):
    filterd_to_explore = []
    avg_reward = {node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 for node in to_explore}

    for node in to_explore:
        if len(childs.get(node,[])) < max_expand or max([avg_reward.get(child,-999) for child in childs.get(node,[])]) < avg_reward.get(node,-999):
            filterd_to_explore.append(node)
    
    return filterd_to_explore


def get_best_explore_from_ucb(to_explore, ucb_bank):
    best_node = None
    highest_ucb = float('-inf')
    
    for node in to_explore:
        ucb_value = ucb_bank.get(node, float('-inf'))
        if ucb_value > highest_ucb:
            highest_ucb = ucb_value
            best_node = node
            
    return best_node

def compute_ucb(r_c, N_n, N_c, C):
    return r_c + C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))

def update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank, C=1.4,gamma=0.85):
    visit_count = {node: len(to_explore_reward[node]) for node in to_explore}

    avg_reward = {node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 for node in to_explore}

    leaves = set(to_explore) - set(fathers.values())
    
    for leaf in leaves:
        # ucb_bank[leaf] = avg_reward[leaf]
        ucb_bank[leaf] = compute_ucb(avg_reward[leaf],len(to_explore_reward.get(fathers.get(leaf,None),[])),len(to_explore_reward.get(leaf,[])),C)
    
    nodes_to_update = list(leaves)
    while nodes_to_update:
        new_nodes_to_update = set()
        for node in nodes_to_update:
            father = fathers.get(node)
            if father is not None:
                if father not in ucb_bank:
                    new_nodes_to_update.add(father)
                if father in ucb_bank:
                    ucb_values = []
                    child_reward = []
                    for child in childs[father]:
                        ucb_values.append(ucb_bank[child])
                        child_reward.append(avg_reward[child])
                    father_reward = (avg_reward[father] + max(child_reward))/2
                    ucb_bank[father] = compute_ucb(father_reward,len(to_explore_reward.get(fathers.get(father,None),[])),len(to_explore_reward.get(father,[])),C)
        nodes_to_update = list(new_nodes_to_update)

def step(query, weak_answer, history=[]):
    print("\tGenerating Feedback from Node")
    hints,history = get_weak_hints(query,weak_answer,history=history)
    print("\tGenerating a better set of SVAs")
    answer,history = get_better_answer(query,weak_answer,hints,history=history)
    return hints,answer,history

def main_loop(query,max_iter=4, historyOG=[]):
    to_explore = []
    to_explore_reward = {}
    history_bank = {}
    hints_bank = {}
    ucb_bank = {}
    fathers = {}
    childs = {}
    def sampling_reward(answer, history):
        if answer not in to_explore_reward:
            to_explore_reward[answer] = []
        reward = cal_reward(query,answer,history)
        to_explore_reward[answer].append(reward)

    def add_to_hints_bank(hints,weak_answer):
        if weak_answer not in hints_bank:
            hints_bank[weak_answer] = []
        hints_bank[weak_answer].append(hints)

    def add_to_childs(father,child):
        if father not in childs:
            childs[father] = []
        childs[father].append(child)

    hints_reward_imp_bank = {}
    def add_to_hints_reward_imp_bank(hints,weak_answer,reward,answer):
        if weak_answer not in hints_reward_imp_bank:
            hints_reward_imp_bank[weak_answer] = []
        hints_reward_imp_bank[weak_answer].append((hints,reward,answer))

    ###get weak answer###
    print("Generating Weak Answer Node")
    weak_answer, history = get_weak_answer(query, historyOG[0])
    history_bank[weak_answer] = tuple(history)
    answers_list = [weak_answer,]
    to_explore = [weak_answer,]
    childs[weak_answer] = []
    fathers[weak_answer] = None
    print("Sampling reward for weak answer")
    sampling_reward(weak_answer, historyOG[1])

    hints_list = []

    update_ucb(fathers=fathers, childs=childs, to_explore=to_explore, to_explore_reward=to_explore_reward, ucb_bank=ucb_bank)

    for i in range(max_iter):
        # Print the rollout number
        print('Iteration:', i)

        # Node Selection
        print("Selecting Best Node and Resampling Reward")
        filterd_to_explore = filter_mature_node(childs, to_explore, to_explore_reward)
        # Update reward calculated for node again
        weak_answer = get_best_explore_from_ucb(filterd_to_explore, ucb_bank)
        sampling_reward(weak_answer, historyOG[1])

        # Node Expansion
        print("Expanding Node")
        hints,answer,history = step(query,weak_answer,history=history_bank[weak_answer])
        add_to_hints_bank(hints,weak_answer)
        history_bank[answer] = tuple(history)
        to_explore.append(answer)

        # Node Evaluation
        print("Node Evaluation")
        sampling_reward(answer, historyOG[1])

        # Auxiliary functions for updating tree
        fathers[answer] = weak_answer
        childs[answer] = []
        add_to_childs(weak_answer,answer)
        answers_list.append(answer)
        hints_list.append(hints)

        # Backpropogation
        print("Backpropogating through the tree\n")
        update_ucb(fathers=fathers,childs=childs,to_explore=to_explore,to_explore_reward=to_explore_reward,ucb_bank=ucb_bank)
        add_to_hints_reward_imp_bank(hints,weak_answer,min(to_explore_reward.get(answer)) - min(to_explore_reward.get(weak_answer)),answer)#ucb_bank[answer] - ucb_bank[weak_answer]

    return hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank

def func(signal_name, signal_information, history, spec_name, document_data):
    query = "Generate SVAs for " + signal_name + ". Here is more information about the signal: " + signal_information
    max_iter = 4

    hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank = main_loop(query, max_iter, history)

    data = {
        'query':query,
        'hints_list':hints_list,
        'answers_list':answers_list,
        'to_explore':to_explore,
        'to_explore_reward':to_explore_reward,
        'hints_bank':hints_bank,
        'history_bank':history_bank,
        'hints_reward_imp_bank':hints_reward_imp_bank,
        'fathers':fathers,
        'childs':childs,
        'ucb_bank':ucb_bank,
    }

    with open(f'./output/json/{spec_name}_{signal_name}.json','w+') as f:
        json.dump(data,f,indent=4,ensure_ascii=False)
    
    # Use the entire data to generate a comprehensive set of SVAs using o1-preview
    client = OpenAI(api_key=os.getenv('API_KEY'))
    client.chat.completions.create(
                model='o1-mini',
                messages=[
                    {"role": "system", "content": COMBINER_PROMPT}
                ],
                temperature=0.95,
                timeout=15
            )
    completion = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {"role": "user", "content": f"Combine all this data {data} along with your own reasoning, and produce a set of COMPLETE, CONSISTENT and CORRECT SVAs. Here is the signal name {signal_name}, all the information from the specification {document_data}, and particular information for the signal: {signal_information}."}
        ],
        temperature=0.95,
    )

    return completion.choices[0].message.content

if __name__ == '__main__':
    directory = "./datasets/info/"
    pdf, spec_name = list_csv_files(directory)
    document_data = read_document(pdf)
    df = pd.read_csv("./datasets/info/" + spec_name + '.csv')

    create_clients()

    history = []
    history.append([document_data + df.loc[0, 'information']])
    history.append([document_data + df.loc[0, 'information']])

    df.loc[1:, 'output'] = df.loc[1:].apply(lambda row: func(row['signal_name'], row['information'], history, row['spec_name'], document_data), axis=1)

    print(df.head())
    
    df.to_csv(f'./output/{spec_name}.csv')
