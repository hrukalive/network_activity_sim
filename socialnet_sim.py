# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
from heapq import heapify, heappush, heappop
from copy import deepcopy
import uuid
import json

#%%
class Message:
    def __init__(self, msg_id, content, referer = None):
        self.id = msg_id
        self.content = content
        self.embedding = np.random.rand(100)
        self.weights = []
        self.history = []
        self.referer = referer

    @staticmethod
    def clone(msg, msg_id, referer = None):
        ret = deepcopy(msg)
        ret.id = msg_id
        ret.referer = referer
        return ret
    
    def addWeight(self, weight):
        self.weights.append(weight)
    def addHistory(self, obj):
        self.history.append(obj)

    def __repr__(self):
        return 'Message({}, {}): '.format(str(self.content), self.id)
    def __str__(self):
        return 'Message({}, {}): '.format(str(self.content), self.id)

#%%
class Event:
    def __init__(self, evt_id, evt_type, sender, receiver, cargo):
        self.id = evt_id
        self.evt_type = evt_type
        self.sender = sender
        self.receiver = receiver
        self.cargo = cargo
    
    def __repr__(self):
        return 'Event({}): {} -> {}'.format(self.evt_type, str(self.sender), str(self.receiver))
    def __str__(self):
        return 'Event({}): {} -> {}'.format(self.evt_type, str(self.sender), str(self.receiver))

class InitiateEvent(Event):
    def __init__(self, evt_id, sender, receiver, cargo):
        super(InitiateEvent, self).__init__(evt_id, 'initiate', sender, receiver, cargo)

class ReactEvent(Event):
    def __init__(self, evt_id, sender, receiver, cargo):
        super(ReactEvent, self).__init__(evt_id, 'react', sender, receiver, cargo)
    def __repr__(self):
        return '{} react to {} from {}'.format(str(self.receiver), str(self.cargo), str(self.sender))
    def __str__(self):
        return '{} react to {} from {}'.format(str(self.receiver), str(self.cargo), str(self.sender))

class NewNodeEvent(Event):
    def __init__(self, evt_id, sender, receiver, cargo):
        super(NewNodeEvent, self).__init__(evt_id, 'new_node', sender, receiver, cargo)

class NewEdgeEvent(Event):
    def __init__(self, evt_id, sender, receiver, cargo):
        super(NewEdgeEvent, self).__init__(evt_id, 'new_edge', sender, receiver, cargo)

#%%
class Agent():
    def __init__(self, agent_id):
        self.id = agent_id
        self.history = []
        self.embedding = np.random.rand(100)

    def reactTime(self):
        return np.random.rand() * 10

    def wouldForward(self, msg):
        return True

    def react(self, world, evt):
        msg = evt.cargo
        if self.wouldForward(msg):
            for n_from, n_to, n_weight in world.graph[self]:
                if n_to != msg.referer:
                    new_msg = Message.clone(msg, uuid.uuid4(), self)
                    new_msg.addWeight(n_weight)
                    new_msg.addHistory(self)
                    world.addEvents(world.t + n_to.reactTime(), ReactEvent(uuid.uuid4(), n_from, n_to, new_msg))
    
    def __repr__(self):
        return 'Agent({})'.format(self.id)
    def __str__(self):
        return 'Agent({})'.format(self.id)

#%%
class SocialWorld:
    def __init__(self):
        self.events = {}
        self.pq = []
        self.graph = {}
        self.agents = {}
        self.t = 0
        self.logs = []
        self.messages = {}

    def setInitialEvents(self, initial_messages):
        for t, evt in initial_messages:
            if t not in self.events:
                self.events[t] = []
            self.events[t].append(evt)
        self.pq = [(t, evts) for t, evts in self.events.items()]
        heapify(self.pq)
    
    def addEvents(self, t, evt):
        if t not in self.events:
            self.events[t] = [evt]
            heappush(self.pq, (t, self.events[t]))
        else:
            self.events[t].append(evt)
    
    def nextEvents(self):
        t, next_evts = heappop(self.pq)
        self.events.pop(t)
        self.t = t
        for evt in next_evts:
            self.process(evt)
    
    def process(self, evt):
        self.logs.append({'t': self.t, 'evt': str(evt)})
        if evt.evt_type == 'initiate':
            msg = evt.cargo
            for n_from, n_to, n_weight in self.graph[evt.receiver]:
                new_msg = Message.clone(msg, uuid.uuid4(), n_from)
                new_msg.addWeight(n_weight)
                new_msg.addHistory(n_from)
                self.addEvents(self.t + n_to.reactTime(), ReactEvent(uuid.uuid4(), n_from, n_to, new_msg))
        elif evt.evt_type == 'react':
            evt.receiver.react(world, evt)
        elif evt.evt_type == 'new_node':
            agent = evt.cargo
            self.agents[agent.id] = agent
        elif evt.evt_type == 'new_edge':
            agent_a, agent_b, n_weight = evt.cargo
            if agent_a not in self.graph:
                self.graph[agent_a] = set()
            self.graph[agent_a].add((agent_a, agent_b, n_weight))
    
    def __repr__(self):
        return 'World'
    def __str__(self):
        return 'World'


#%%
world = SocialWorld()

#%%
agents = [
    Agent('0'),
    Agent('1'),
    Agent('2'),
    Agent('3')
]

#%%
init_evts = [
    (0, NewNodeEvent(uuid.uuid4(), None, world, agents[0])),
    (0, NewNodeEvent(uuid.uuid4(), None, world, agents[1])),
    (0, NewNodeEvent(uuid.uuid4(), None, world, agents[2])),
    (0, NewNodeEvent(uuid.uuid4(), None, world, agents[3])),
    (0, NewEdgeEvent(uuid.uuid4(), None, world, (agents[0], agents[1], 0.1))),
    (0, NewEdgeEvent(uuid.uuid4(), None, world, (agents[1], agents[2], 0.2))),
    (0, NewEdgeEvent(uuid.uuid4(), None, world, (agents[2], agents[3], 0.3))),
    (0, NewEdgeEvent(uuid.uuid4(), None, world, (agents[3], agents[1], 0.4))),
    (0, NewEdgeEvent(uuid.uuid4(), None, world, (agents[3], agents[2], 0.5))),
    (0, NewEdgeEvent(uuid.uuid4(), None, world, (agents[1], agents[0], 0.6))),
    (1, InitiateEvent(uuid.uuid4(), world, agents[0], Message(uuid.uuid4(), 'm1'))),
    (1, InitiateEvent(uuid.uuid4(), world, agents[1], Message(uuid.uuid4(), 'm2'))),
    (2, InitiateEvent(uuid.uuid4(), world, agents[0], Message(uuid.uuid4(), 'm3'))),
    (3, InitiateEvent(uuid.uuid4(), world, agents[0], Message(uuid.uuid4(), 'm4'))),
]

#%%
world.setInitialEvents(init_evts)

#%%
# world.nextEvents()
# print('\n'.join(['{}:\n\t{}\n'.format(t, '\n\t'.join([str(e) for e in world.events[t]])) for t in sorted(world.events.keys())]))

while world.t < 100:
    world.nextEvents()
    print(world.t)
with open('logs.txt', 'w') as f:
    for log in world.logs:
        f.write(json.dumps(log) + '\n')

#%%
for a, b in world.graph.items():
    print(str(a), '->', ' '.join([str((str(c[0]), str(c[1]), c[2])) for c in b]))

#%%
