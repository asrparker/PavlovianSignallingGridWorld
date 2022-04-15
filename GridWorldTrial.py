import learn
import time
import datetime

import csv
import numpy as np

from dynamic_plotterm import *

# control variables
index = 4
indexLast = 4
x = 0
indexIncrease = True

threshold = 0.75

c = 0
clast = 0

upticks = 0
dwnticks = 0
fup = 0
fdwn = 0

upC = 0
downC = 0
timeSince_upC = 0
timeSince_downC = 0
upTime = 0
downTime = 0

cycle = 1

direction_size = 32
world_size = 3 * direction_size

upper_contact = 31
lower_contact = 0

# learning variables
gamma = 0.9
lamb = 0.9
alpha_W = 0.2  # *(1-lamb)
alpha_omega = 0.01

feature = 0

# set state once to get vector size
S = learn.manage_state(world_size)
S.update_state(0)
state_size = len(S.state_vector)

agent_up = learn.GTDGVF(gam=gamma, lam=lamb, alpha_W=alpha_W, alpha_omega=alpha_omega, size=state_size)
agent_down = learn.GTDGVF(gam=gamma, lam=lamb, alpha_W=alpha_W, alpha_omega=alpha_omega, size=state_size)
agent_TD = learn.GTDGVF(gam=gamma, lam=lamb, alpha_W=0.1, alpha_omega=0, size=state_size)

# observe = DynamicPlot(window_x=50, title='Offline Pavlovian Control Testing', xlabel='Time Step', ylabel='Value',
#                       num_plots=3)
# observe.add_line('Command', 0)
# observe.add_line('Prediction Up', 1)
# observe.add_line('Prediction Down', 1)
# observe.add_line('Learning Error Up', 2)
# observe.add_line('Learning Error Down', 2)


# open the file in the write mode
f = open('C:/Adam/Experiment Data/Pavlovian Signaling/Trial Data/Sim/sim.csv', 'wb')

# create the csv writer
writer = csv.writer(f)
writer.writerow(['Cycle Number', 'Index', 'Contact', 'Prediction'])

start = time.time()
print(datetime.datetime.now())

while cycle <= 100:
    # get command
    indexLast = index

    # if agent_TD.predict > 0.75 or index == upper_contact:
    #     indexIncrease = False
    # elif agent_TD.predict > 0.75 or index == lower_contact:
    #     indexIncrease = True
    if agent_up.predict > threshold or index == upper_contact:
        indexIncrease = False
        cycle += 1
    elif agent_down.predict > threshold or index == lower_contact:
        indexIncrease = True
        cycle += 1

    if indexIncrease is True and index < upper_contact:
        index += 1
    elif indexIncrease is False and index > lower_contact:
        index -= 1

    if index == upper_contact or index == lower_contact:
        c = 1
        contact = 1
    else:
        c = 0
        contact = 0

    # if agent_TD.predict > 0.75:
    #     prediction = 1
    if agent_up.predict > threshold or agent_down.predict > threshold:
        prediction = 1
    else:
        prediction = 0

    if index == upper_contact:
        upC += 1
        timeSince_upC = time.time() - upTime
        upTime = time.time()
        # print(index, upC, timeSince_upC)
        # print(datetime.datetime.now())
    elif index == lower_contact:
        downC += 1
        timeSince_downC = time.time() - downTime
        downTime = time.time()
        # print(index, downC, timeSince_downC)
        # print(datetime.datetime.now())

    ac = abs(c)

    # find active feature
    feature = index
    # move index to a higher part of the space for moving down or being still
    if index < indexLast:
        feature += direction_size
    elif index == indexLast:
        feature += 2 * direction_size
    S.update_state(int(feature))

    agent_TD.update(c, S.state_vector, rho=1)
    if (feature >= 0) and (feature < direction_size):
        agent_up.update(c, S.state_vector, rho=1)
        agent_down.update(c, S.state_vector, rho=0)
    elif (feature >= direction_size) and (feature < direction_size * 2):
        agent_up.update(c, S.state_vector, rho=0)
        agent_down.update(c, S.state_vector, rho=1)
    else:
        agent_up.update(c, S.state_vector, rho=0)
        agent_down.update(c, S.state_vector, rho=0)

    # observe.update(x, [index, agent_up.predict, agent_down.predict, agent_up.delta, agent_down.delta])
    # time.sleep(.000001)

    writer.writerow([cycle, index, contact, prediction])

    x += 1


# close the file
f.close()
