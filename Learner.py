import random
import sys
import akshaynet
import bisect
import netTheano as nn

import numpy as np
import c4board as bb

class Learner:
    # All values are with respect to player 1 ('X')
    values = {}
    occurrences = {}
    currNet = None
    prevNet = None

    def generateNet(self, epochs, lr):
        boards = []
        vals = []

        for b, v in self.values.items():
            boards.append(bb.convertNP(b))
            vals.append(v)
        x = np.array(boards)
        y = np.array(vals)
        print len(x)
        sizes = [42, 36,  24, 8, 1]
        self.currNet = akshaynet.Network(sizes, 1)

        print self.currNet.evaluate(zip(x, y))
        self.currNet.SGD(zip(x, y), epochs, 50, float(lr))
        print self.currNet.evaluate(zip(x, y))
        self.prevNet = self.currNet.copy()

    def genTheanoNet(self, epochs, lr):
        boards = []
        vals = []

        for b,v in self.values.items():
            boards.append(bb.convertGrid(b))
            vals.append(v)
        s = np.array(boards)
        y = np.array(vals)
        data = zip(s,y)
        random.shuffle(data)
        train_data = data[0:int(.7*len(data))]
        val_data = data[int(.7*len(data)): int(.85 * len(data))]
        test_data = data[int(.85*len(data)):]

        mini_batch_size = 50
        
        self.currNet = nn.Network([nn.ConvPoolLayer(image_shape=(mini_batch_size, 1, 7, 6), 
                      filter_shape=(20, 1, 2, 2), 
                      poolsize=(2, 2)),
        nn.FullyConnectedLayer(n_in=20*6*5, n_out=40),
        nn.SoftmaxLayer(n_in=40, n_out=1)], mini_batch_size)
        self.currNet.SGD(train_data, epochs, mini_batch_size, lr, val_data, test_data)
        self.prevNet = self.currNet
        #THE ABOVE SHOULD BE A SEPARATE COPY EVENTUALLY

    def learnExperts(self, num = 100000):
        with open('connect-4.data') as f:
            content = f.readlines()
        for i in range(0, min(num, len(content))):
            line = content[i]
            info = line.split(',')
            b = bb.Board()
            for col in range(0, 7):
                for row in range(0, 6):
                    char = info[col * 6 + row]
                    if char == 'o':
                        b.rows[col][row] = 'O'
                    elif char == 'x':
                        b.rows[col][row] = 'X'
                    elif char == 'b':
                        b.rows[col][row] = '.'
                    else:
                        print 'didnt recognize char'
            self.occurrences[b.convert()] = 1
            if info[42] == 'win\n':
                self.values[b.convert()] = .7 + random.gauss(0, .08)

            elif info[42] == 'loss\n':
                self.values[b.convert()] = .3 + random.gauss(0, .08)
            elif info[42] == 'draw\n':
                self.values[b.convert()] = .5 + random.gauss(0, .08)
            else:
                print 'didnt recognize result...'

    def searchValue(self, board, player, depth):
        if board.won():
            # returns 0 if player 1's turn, vice versa
            return player - 1

        if board.full():
            return .5
        if depth == 0:
            return self.currNet.feedforward(bb.convertNP(board.convert()))
        else:
            slots = board.openSlots()
            if player == 1:
                return max(self.searchValue(board.copy().place(slot, 'X'), 2, depth - 1) for slot in slots)
            else:
                return min(self.searchValue(board.copy().place(slot, 'O'), 1, depth - 1) for slot in slots)

    def chooseMoveSearch(self, board, player):
        move = 0
        maxVal = player - 1
        slots = board.openSlots()
        if len(slots) == 1:
            return slots[0]
        if player == 1:
            for slot in slots:
                val = self.searchValue(board.copy().place(slot, 'X'), 2, 4)
                if val > maxVal:
                    maxVal = val
                    move = slot
        else:
            for slot in slots:
                val = self.searchValue(board.copy().place(slot, 'O'), 1, 4)
                if val < maxVal:
                    maxVal = val
                    move = slot
        if maxVal == .5:
            print '?'
        #x = random.randint(1,10)
        print 'move is '
        print move
        if move == 0:
            return slots[random.randint(0, len(slots) - 1)]
        else:
            return move

    def chooseMoveMC(self, board, player):
        move = 0
        move2 = 0
        maxVal = 0
        smaxVal = 0
        chosen = False
        if player == 1:

            char = 'X'
        else:
            maxVal = 1
            smaxVal = 0
            char = 'O'
        slots = board.openSlots()

        if len(slots) == 1:
            # print 'only one move'
            return slots[0]
        for slot in slots:
            tempBoard = board.copy()
            tempBoard.place(slot, char)
            if tempBoard.won():
                chosen = True
                return slot

            elif tempBoard.convert() in self.values:
                temp_val = float(self.values[tempBoard.convert(
                )]) + random.gauss(0, .08) / self.occurrences[tempBoard.convert()]
                if player == 1:
                    if temp_val > smaxVal:
                        if temp_val > maxVal:
                            maxVal = temp_val
                            move = slot
                            chosen = True
                        else:
                            smaxVal = temp_val
                            move2 = slot
                            chosen = True
                else:
                    if temp_val < smaxVal:
                        if temp_val < maxVal:
                            maxVal = temp_val
                            move = slot
                            chosen = True
                        else:
                            smaxVal = temp_val
                            move2 = slot
                            chosen = True
            else:
                temp_val = .5 + random.gauss(0, .06)
                if player == 1:
                    if temp_val > smaxVal:
                        if temp_val > maxVal:
                            maxVal = temp_val
                            move = slot
                            chosen = True
                        else:
                            smaxVal = temp_val
                            move2 = slot
                            chosen = True
                else:
                    if temp_val < smaxVal:
                        if temp_val < maxVal:
                            maxVal = temp_val
                            move = slot
                            chosen = True
                        else:
                            smaxVal = temp_val
                            move2 = slot
                            chosen = True

        x = random.randint(1, 100)

        if x < 6 or not chosen:
            return slots[random.randint(0, len(slots) - 1)]
        else:
            # return move 1 or move 2
            threshold = maxVal / (maxVal + smaxVal)
            if random.random() < threshold:
                return move
            else:
                return move2


    def playSelf2(self):
        boardRows = []
        board = bb.Board()
        board.initialize()
        # print 'board initialized'
        # board.printBoard()
        result = 0
        while True:
            # Player 1
            if board.full():
                break
            boardRows.append(board.convert())
            move = self.chooseMoveMC(board, 1)
            board.place(move, 'X')
            # board.printBoard()
            if board.won():
                result = 1
                break
            if board.full():
                result = 0
                break

            # Player 2
            boardRows.append(board.convert())
            move = self.chooseMoveMC(board, 2)
            board.place(move, 'O')
            # board.printBoard()
            if board.won():
                result = -1
                break
            if board.full():
                result = 0
                break

        self.values[boardRows[len(boardRows) - 1]] = (result + 1) / 2
        if boardRows[len(boardRows) - 1] in self.occurrences:
            self.occurrences[boardRows[len(boardRows) - 1]] += 1
        else:
            self.occurrences[boardRows[len(boardRows) - 1]] = 1
        for i in range(2, len(boardRows) + 1):

            if boardRows[len(boardRows) - i] in self.values:
                self.values[boardRows[len(boardRows) - i]] = self.values[boardRows[len(boardRows) - i]] + .2 * (
                    self.values[boardRows[len(boardRows) - (i - 1)]] - self.values[boardRows[len(boardRows) - i]])
                self.occurrences[boardRows[
                    len(boardRows) - i]] = self.occurrences[boardRows[len(boardRows) - i]] + 1
            else:
                self.values[boardRows[
                    len(boardRows) - i]] = self.values[boardRows[len(boardRows) - (i - 1)]] - .1 * (self.values[boardRows[len(boardRows) - (i - 1)]] - .5)
                self.occurrences[boardRows[len(boardRows) - i]] = 1

    def playSelfNet(self):
        boardRows = []
        
        board = bb.Board()
        board.initialize()
        # print 'board initialized'
        # board.printBoard()
        result = 0
        x = random.random()

        if x < .5:
            while True:
                # Player 1
                if board.full():
                    break
                boardRows.append(board.convert())
                move = self.pickBestMoveCurrNet(board, 1)
                board.place(move, 'X')
                # board.printBoard()
                if board.won():
                    result = 1
                    break
                if board.full():
                    result = 1
                    break

                # Player 2
                boardRows.append(board.convert())
                move = self.pickBestMovePrevNet(board, 2)
                board.place(move, 'O')
                # board.printBoard()
                if board.won():
                    result = -1
                    break
                if board.full():
                    result = 0
                    break
        else:
            while True:
                # Player 1
                if board.full():
                    break
                # Player 2
                boardRows.append(board.convert())
                move = self.pickBestMovePrevNet(board, 1)
                board.place(move, 'X')
                # board.printBoard()
                if board.won():
                    result = -1
                if board.full():
                    result = 0

                boardRows.append(board.convert())
                move = self.pickBestMoveCurrNet(board, 2)
                board.place(move, 'O')
                # board.printBoard()
                if board.won():
                    result = 1
                    break
                if board.full():
                    result = 0

        self.values[boardRows[len(boardRows) - 1]] = (result + 1) / 2
        if boardRows[len(boardRows) - 1] in self.occurrences:
            self.occurrences[boardRows[len(boardRows) - 1]] += 1
        else:
            self.occurrences[boardRows[len(boardRows) - 1]] = 1
        for i in range(2, len(boardRows) + 1):

            if boardRows[len(boardRows) - i] in self.values:
                self.values[boardRows[len(boardRows) - i]] = self.values[boardRows[len(boardRows) - i]] + .2 * (
                    self.values[boardRows[len(boardRows) - (i - 1)]] - self.values[boardRows[len(boardRows) - i]])
                self.occurrences[boardRows[
                    len(boardRows) - i]] = self.occurrences[boardRows[len(boardRows) - i]] + 1
            else:
                self.values[boardRows[
                    len(boardRows) - i]] = self.values[boardRows[len(boardRows) - (i - 1)]] - .1 * (self.values[boardRows[len(boardRows) - (i - 1)]] - .5)
                self.occurrences[boardRows[len(boardRows) - i]] = 1
        return (result + 1) / 2

    def test(self):
        boardRows = []
        
        board = bb.Board()
        board.initialize()
        # print 'board initialized'
        # board.printBoard()
        results = [0,0,0,0]
        x = random.random()

        if x < .5:
            while True:
                # Player 1
                if board.full():
                    break
                boardRows.append(board.convert())
                move = self.pickBestMoveCurrNet(board, 1)
                board.place(move, 'X')
                # board.printBoard()
                if board.won():
                    results[0] += 1
                    break
                if board.full():
                    results[0] += .5
                    results[1] += .5
                    break

                # Player 2
                boardRows.append(board.convert())
                #change the below back to prev
                move = self.pickBestMovePrevNet(board, 2)
                board.place(move, 'O')
                # board.printBoard()
                if board.won():
                    results[1]+=1
                    break
                if board.full():
                    results[0] += .5
                    results[1] += .5
                    break
            return results
        else:
            while True:
                # Player 1
                if board.full():
                    break
                # Player 2
                boardRows.append(board.convert())
                #change back to prev
                move = self.pickBestMovePrevNet(board, 1)
                board.place(move, 'X')
                # board.printBoard()
                if board.won():
                    results[3] += 1
                    break
                if board.full():
                    results[2] += .5
                    results[3] += .5
                    break

                boardRows.append(board.convert())
                move = self.pickBestMoveCurrNet(board, 2)
                board.place(move, 'O')
                # board.printBoard()
                if board.won():
                    results[2]+= 1
                    break
                if board.full():
                    results[2] += .5
                    results[3] += .5
                    break
            return results



        
        

    def learn(self, iterations):
        for i in range(0, iterations):
            # print i
            if i % 1000 == 0:
                print i
            self.playSelf2()



    def learnNet(self, iterations, epochs, lr):
        v = {}
        o = {}
        for x,y in self.values.items():
            if random.random() > .7:
                v[x] = y
                o[x] = self.occurrences[x]
        self.values = v
        self.occurrences = o
        
        wins = 0
        for i in range(0,iterations):
            if i%100 == 0:
                print i
                print len(self.values)
                
            wins += self.playSelfNet()

        boards = []
        vals = []
        print 'wins: '
        print wins

        for b, v in self.values.items():
            boards.append(bb.convertNP(b))
            vals.append(v)

        x = np.array(boards)
        y = np.array(vals)
        print self.currNet.evaluate(zip(x, y))

        self.prevNet = self.currNet.copy()
        self.currNet.SGD(zip(x, y), epochs, 50, float(lr))
        print self.currNet.evaluate(zip(x, y))

    def testNets(self, iterations):
        self.values = {}
        self.occurrences = {}
        results = [0,0,0,0]
        for i in range(0,iterations):
            if i%100 == 0:
                print i
                print len(self.values)
                
            r  = self.test()
            for i in range(0,len(results)):
                results[i] += r[i]

        boards = []
        vals = []
        print results



    def pickBestMoveCurrNet(self, board, player):
        move = 0
        maxVal = player - 1
        slots = board.openSlots()
        slotVals = []
        if len(slots) == 1:
            return slots[0]
        if player == 1:
            for slot in slots:
                slotVals.append(float(self.currNet.feedforward(bb.convertNP(board.copy().place(slot, 'X').convert()))))
                
        else:
            for slot in slots:
                slotVals.append(1-float(self.currNet.feedforward(bb.convertNP(board.copy().place(slot, 'O').convert()))))
        for i in range(0, len(slotVals)):
            slotVals[i] = slotVals[i] ** 3
        s = sum(slotVals)
        if s == 0:
            return slots[0]
        for i in range(0,len(slotVals)):
            slotVals[i] = slotVals[i] / s
        s = 0        
        cdf = []

        for i in range(0,len(slotVals)):
            cdf.append(s + slotVals[i])
            s += slotVals[i]
        #print cdf
        move = slots[bisect.bisect(cdf,random.random())]
        #print move
        return move

    
    def pickBestMovePrevNet(self, board, player):
        
        move = 0
        maxVal = player - 1
        slots = board.openSlots()
        slotVals = []
        if len(slots) == 1:
            return slots[0]
        if player == 1:
            for slot in slots:
                slotVals.append(float(self.prevNet.feedforward(bb.convertNP(board.copy().place(slot, 'X').convert()))))
                
        else:
            for slot in slots:
                slotVals.append(1-float(self.prevNet.feedforward(bb.convertNP(board.copy().place(slot, 'O').convert()))))
        for i in range(0, len(slotVals)):
            slotVals[i] = slotVals[i] ** 3
        s = sum(slotVals)
        if s == 0:
            return slots[0]
        for i in range(0,len(slotVals)):
            slotVals[i] = slotVals[i] / s
        s = 0        
        cdf = []

        for i in range(0,len(slotVals)):
            cdf.append(s + slotVals[i])
            s += slotVals[i]
        #print cdf
        move = slots[bisect.bisect(cdf,random.random())]
        #print move
        return move
        



