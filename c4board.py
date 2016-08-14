# board always contains gamestate
# checks where player can put stuff, checks if someone won
# has a nice print function
import random
import sys
import akshaynet
import bisect

import numpy as np


def listGreaterEqual(l1, l2):
    for i in range(0, len(l1)):
        if l1[i] > l2[i]:
            return True
        elif l2[i] > l1[i]:
            return False
    return True


def convertNP(s):
    x = []

    for i in s:
        if i == '.':
            x.append(0)
        elif i == 'O':
            x.append(-1)
        elif i == 'X':
            x.append(1)
        else:
            print 'problem converting to np array'
    retval = np.random.randn(42, 1)
    for i in range(0, 42):
        retval[i] = float(x[i])
    return retval

def convertGrid(s):
    x = np.full((7,6), 0.0)
    vals = convertNP(s)
    for i in range(0,42):
        x[i / 6][i % 6] = float(vals[i][0])
    return x


class Board:
    rows = [['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], [
        '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.']]

    def place(self, move, character):
        if move < 1 or move > 7:
            print move
            print "invalid move; must be 1-7"
        else:
            i = 0
            while i < 6 and self.rows[move - 1][i] != '.':
                i = i + 1
            self.rows[move - 1][i] = character
        return self

    def initialize(self):
        self.rows = [['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], [
            '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.']]

    def convert(self):
        retval = []
        retval2 = []
        for i in range(0, 7):
            for j in range(0, 6):
                retval.append(self.rows[i][j])
        tempBoard = self.copy()
        tempBoard.rows.reverse()
        for i in range(0, 7):
            for j in range(0, 6):
                retval2.append(tempBoard.rows[i][j])
        if listGreaterEqual(retval, retval2):
            return ''.join(retval)

        return ''.join(retval2)

    def copy(self):
        newboard = Board()
        r = [['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], [
            '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.']]
        for i in range(0, 7):
            for j in range(0, 6):
                r[i][j] = self.rows[i][j]
        newboard.rows = r
        return newboard

    def won(self):

        # check horizontal spaces
        for y in range(6):
            for x in range(4):
                if self.rows[x][y] == self.rows[x + 1][y] == self.rows[x + 2][y] == self.rows[x + 3][y]:
                    if self.rows[x][y] != '.':
                        return True

            # check vertical spaces
        for x in range(7):
            for y in range(3):
                if self.rows[x][y] == self.rows[x][y + 1] == self.rows[x][y + 2] == self.rows[x][y + 3]:
                    if self.rows[x][y] != '.':
                        return True

            # check / diagonal spaces
        for x in range(7 - 3):
            for y in range(3, 6):
                if self.rows[x][y] == self.rows[x + 1][y - 1] == self.rows[x + 2][y - 2] == self.rows[x + 3][y - 3]:
                    if self.rows[x][y] != '.':
                        return True

            # check \ diagonal spaces
        for x in range(7 - 3):
            for y in range(6 - 3):
                if self.rows[x][y] == self.rows[x + 1][y + 1] == self.rows[x + 2][y + 2] == self.rows[x + 3][y + 3]:
                    if self.rows[x][y] != '.':
                        return True

        return False

    def full(self):
        for i in range(0, 7):
            for j in range(0, 6):
                if self.rows[i][j] == '.':
                    return False
        return True
    def printBoard(self):
        # Can be implemented more cleanly
        # print len(self.rows[0])
        print ''
        for i in range(0, 6):
            print ''
            for j in range(0, 7):
                sys.stdout.write(' ')
                sys.stdout.write(self.rows[j][5 - (i)])
        print ''
    def openSlots(self):
        slots = []
        for i in range(0, 7):
            if self.rows[i][5] == '.':
                slots.append(i + 1)
        return slots



