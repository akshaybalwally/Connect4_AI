import random
import sys
import akshaynet
import bisect
import c4board as bb
import Learner as ll

import numpy as np

def playGame(l):

    board = bb.Board()
    board.initialize()
    board.printBoard()

    g = random.random()
    if g < .5:

        while (True):
            board.printBoard()
            print 'AI to move'
            p1move = l.chooseMoveSearch(board, 1)
            board.place(p1move, 'X')

            if board.won():
                print 'Player 1 won'
                board.printBoard()
                break
            elif board.full():
                print 'tie'
                break

            board.printBoard()
            print 'open spots: '
            print board.openSlots()

            p2move = input('Humans move\n')
            board.place(p2move, 'O')

            if board.won():
                board.printBoard()
                print 'Player 2 won'
                break
            elif board.full():
                print 'tie'
                break
    else:
         while (True):

            board.printBoard()
            print 'open spots: '
            print board.openSlots()

            p2move = input('Humans move\n')
            board.place(p2move, 'X')

            if board.won():
                board.printBoard()
                print 'Player 2 won'
                break
            elif board.full():
                print 'tie'
                break


            board.printBoard()
            print 'AI to move'
            p1move = l.chooseMoveSearch(board, 2)
            board.place(p1move, 'O')

            if board.won():
                print 'Player 1 won'
                board.printBoard()
                break
            elif board.full():
                print 'tie'
                break

def makeLearner():
    l = ll.Learner()
    l.learnExperts('connect-4.data')
    l.learn(100)

    boards = []
    vals = []

    for b, v in l.values.items():
        boards.append(bb.convertNP(b))
        vals.append(v)
    x = np.array(boards)
    y = np.array(vals)

    sizes = [42, 28, 8, 1]
    n = akshaynet.Network(sizes)
    n.SGD(zip(x, y), 100, 1000, 1)

    return l
print 'beginning journey'
l = ll.Learner()
l.learnExperts(10000)

# print 'learned experts'
l.learn(5000)
# print 'learned'
# print len(l.values)
#l.learn(2000)
l.generateNet(30, .03)
#l.genTheanoNet(25,.03)
l.learnNet(1000, 25, .02)
l.learnNet(1000, 25, .02)
l.learnNet(1000, 25, .02)
l.learnNet(1000, 25, .02)
# l.learnNet(1000, 25, .02)
# l.learnNet(1000, 25, .02)
# l.learnNet(8000, 25, .02)
l.learnNet(4000,40,.02)


#l.learnNet(80,25,.04)
print l.currNet.equals(l.prevNet)

while(True):
    playGame(l)
# playGame(l)
# print 'generated net'
# l.learnNet(15000, 30)
# print 'learned net once'
# l.learnNet(15000,20)
# print 'learned net twice'
# l.learnNet(15000,10)
# print 'learned net thrice'
# while(True):
#     playGame(l)
