# from collections import deque, ChainMap, Counter
# import functions
from random import randint
# import matplotlib.pyplot as plt 
import math, statistics
# import sys
import time
import os
import random
from itertools import combinations
import collections
# from collections import Counter
# import copy
# import heapq
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
from typing import List, Optional

# 3d array [a][b][c] : a the depth, b rows and c columns

############

def sortColors(nums):
  """
  Do not return anything, modify nums in-place instead.
  """
  l = len(nums)
  if l == 1: return
  elif l == 2:
    if nums[0] > nums[1]:
      nums[0], nums[1] = nums[1], nums[0]
    else: return
  #
  def _swap_(i1, i2):
    nums[i1], nums[i2] = nums[i2], nums[i1]
  #
  for _ in range(l - 2):
    if nums[_] > nums[_ + 1]: _swap_(_, _ + 1)
    if nums[_ + 1] > nums[_ + 2]: _swap_(_ + 1, _ + 2)
    if nums[_] > nums[_ + 1]: _swap_(_, _ + 1)

def moveZeroes(nums):
  """
  Do not return anything, modify nums in-place instead.
  """
  def _swap_(i1, i2):
    nums[i1], nums[i2] = nums[i2], nums[i1]

  l = len(nums)
  # for i in range(l):
  #   a = i
  #   if nums[i] == 0:
  #     while a != l:
  #       if nums[a] != 0: # first non-zero element encounterd
  #         _swap_(i, a)
  #         break
  #       a += 1
  #     else: return

  c = 0 # number of 0s
  index = 0
  while index < l:
    print(c, index)
    a = 0
    if nums[index] == 0: c += 1
    else:
      for i in range(c):
        if (index + i) == l: return
        _swap_(index + i, index - c + i)
        a += 1
      index += a
      continue
    index += + 1

class ListNode:
  def __init__(self):
    return
  
class TreeNode:
  def __init__(self):
    return
# nums = [0,1,0,3,12,0,4]
# # nums = [1,0]
# moveZeroes(nums)
# print(nums)  

def trap(height): # height = [nums]
  a = 0 # area
  la = 0 # last area
  w = height[0] # wall
  sw = 0 # subwall
  i = 0 # index if wall

  for _ in range(len(height)):
    if height[_] < w:
      t = height[_]*(_ - i - 1) - sw
      a += t
      la += t
      sw += height[_]
    elif height[_] >= w:
      t = w*(_ - i - 1) - sw
      a += (t - la)
      la = 0
      w = height[_]
      i = _
      sw = 0
    print(f"a: {a} | t: {t} | w: {w} | i: {i} | _: {_} | sw: {sw}")
  
  return a

def hasCycle(head):
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  n = {}
  while head:
    if head not in n:
      n[head] = 0
    else:
      return False
    head = head.next
  return True

def isSymmetric(root):
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, val=0, left=None, right=None):
  #         self.val = val
  #         self.left = left
  #         self.right = right
  def _run_(l = root.left, r = root.right):
    print(l.val if l else l, r.val if r else r)
    if l == None and r == None:
      return True
    if (l != None) ^ (r != None):
      return False
    if l.val != r.val:
      return False
    else:
      return _run_(l.left, r.right) and _run_(l.right, r.left)
  
  return _run_()

def merge(nums1, m, nums2, n):
  """
  Do not return anything, modify nums1 in-place instead.
  """
  nums1 += nums2
  nums1.sort()

def maximumHappinessSum(happiness, k):
  happiness.sort()
  r = 0
  for _ in range(k):
    t = happiness[-1 - _] - _
    print(t)
    if t <= 0: break
    else:
      r += t
  return r

def getDescentPeriods(prices):
  dp = 1 # descent periods
  r = 0
  l = len(prices)
  if l == 1: return 1
  for i in range(l - 1): # i up to n - 2
    print(dp, r, prices[i])
    if i == (l - 2):
      if prices[i] == (prices[i + 1] + 1):
        for _ in range(1, dp + 2):
          r += math.comb(dp + 1, _)
      else:
        for _ in range(1, dp + 1):
          r += math.comb(dp, _)
        r += 1
    if prices[i] == (prices[i + 1] + 1):
      dp += 1
    else:
      for _ in range(1, dp + 1):
        r += math.comb(dp, _)
      dp = 1
  return r

# def sortedArrayToBST(nums):
#   # Definition for a binary tree node.
#   class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#       self.val = val
#       self.left = left
#       self.right = right
#   #
#   for

#   return rs

# print(sortedArrayToBST([-10,-3,0,5,9]))

def minDepth(root):
  if not root: return 0
  def _run_(d = 1, node = root):
    if (node.left == None and node.right == None):
      return d
    elif node.left and node.right:
      return min(_run_(d + 1, node.left), _run_(d + 1, node.right))
    if node.left:
      return _run_(d + 1, node.left)
    else:
      return _run_(d + 1, node.right)
  
  return _run_()

def preorderTraversal(root):
  rs = []

  def _run_(node = root):
    if not node: return
    rs.append(node.val)
    if not node.left:
      _run_(node.left)
    if not node.right:
      _run_(node.right)
  _run_()
  return rs

def majorityElement(nums):
  d = {}
  l = len(nums) 
  for _ in nums:
    if _ not in d: d[_] = 1
    else:
      d[_] += 1
      if d[_] >= l/2:
        return d[_]
      
def uniquePaths(m, n):
  i = [0]
  def _run_(r = 1, c = 1):
    if r == m and c == n:
      i[0] += 1
      return
    if r != m and c != n:
      _run_(r + 1, c)
      _run_(r, c + 1)
      return
    if r == m:
      _run_(r, c + 1)
    if c == n:
      _run_(r + 1, c)

  _run_()
  return i[0]

def threeSum(nums):
  nums.sort()
  l = len(nums)
  rs = []

  i = 0
  while i < l:
    if i >= 1:
      while i < l - 1 and nums[i] == nums[i - 1]:
        i += 1
    lf = i + 1
    r = l - 1
    while lf < r:
      s = nums[lf] + nums[r]
      if s == -nums[i]:
        rs.append([nums[i], nums[lf], nums[r]])
        lf += 1
        while lf < l and nums[lf] == nums[lf - 1]: lf += 1
        r -= 1
      elif s < -nums[i]:
        lf += 1
        while lf < l and nums[lf] == nums[lf - 1]: lf += 1
      else: # s > -nums[i]
        r -= 1
    i += 1
  return rs

def threeSumClosest(nums, target):
  nLen = len(nums)
  nums.sort()
  print(nums)
  result = nums[0] + nums[1] + nums[2]

  index = 0
  while index < nLen:
    if index >= 1:
      while index < nLen and nums[index] == nums[index - 1]:
        index += 1
    #
    # 2 pointers
    left, right = index + 1, nLen - 1
    while left < right:
      print(index, left, right)
      sm = nums[left] + nums[right] + nums[index]
      if sm == target:
        return sm
      else:
        if abs(target - sm) < abs(target - result):
          result = sm
      if sm < target:
        left += 1
        while left < nLen and nums[left] == nums[left - 1]:
          left += 1
      else:
        right -= 1
    #
    index += 1
  return result

def multiply(num1, num2):
  n1, n2 = 0, 0
  l1, l2 = len(num1), len(num2)
  i1, i2 = 0, 0
  
  for i in range(min(l1, l2)):
    n1 = n1*10 + int(num1[i1])
    n2 = n2*10 + int(num2[i2])
    i1 += 1
    i2 += 1
  while i1 != l1:
    n1 = n1*10 + int(num1[i1])
    i1 += 1
  while i2 != l2:
    n2 = n2*10 + int(num2[i2])
  return f"{n1*n2}"

def setZeroes(matrix):
  """
  Do not return anything, modify matrix in-place instead.
  """
  nRow = len(matrix)
  nCol = len(matrix[0])
  for row in range(nRow):
    for column in range(nCol):
      if matrix[row][column] == 0:
        matrix[0][column] = 0
        matrix[row][0] = 0

  for row in range(nRow):
    print(matrix[row])

  for col in range(nCol):
    if matrix[0][col] == 0:
      for i in range(nRow):
        matrix[i][col] = 0

  for row in range(nRow):
    if matrix[row][0] == 0:
      for i in range(nCol):
        matrix[row][i] = 0

  print("------")
  for row in range(nRow):
    print(matrix[row])

def isMatch(s, p): # have not completed yet
  def _run_(s = s, p = p, sIndex = len(s) - 1, pIndex = len(p) - 1):
    while True:
      print(sIndex, pIndex)
      if pIndex < 0 and sIndex < 0: return True
      if pIndex < 0 and sIndex >= 0: return False
      if pIndex >= 0 and sIndex < 0:
        if pIndex % 2 == 0: return False
        for i in range(1, pIndex + 1, 2):
          if p[i] == "*":
            continue
          else: return False
        else: return True
      #
      if p[pIndex] == '.':
        sIndex -= 1
        pIndex -= 1
        continue
      if p[pIndex] == '*':
        pIndex -= 1
        if p[pIndex] == '.':
          if pIndex == 0: return True
          else:
            pIndex -= 1
            while p[pIndex] == '*':
              pIndex -= 2
            for i in range(sIndex + 1):
              if s[i] == p[pIndex]:
                if _run_(s, p, i, pIndex):
                  return True
                else: continue
            else:
              return False
        else:
          while sIndex >= 0 and s[sIndex] == p[pIndex]:
            sIndex -= 1
            if _run_(s, p, sIndex, pIndex - 1):
              return True
            else: continue
        #
        pIndex -= 1
        continue
      if p[pIndex] == s[sIndex]:
        sIndex -= 1
        pIndex -= 1
        continue
      else:
        return False
  return _run_()
      
def fourSum(nums, target):
  if len(nums) <= 3: return []
  nums.sort()
  print(nums)
  result = []
  i1 = 0
  i2 = 1

  while i1 < len(nums) - 2:
    if i1 >= 1:
      while i1 < (len(nums) - 2) and nums[i1] == nums[i1 - 1]:
        i1 += 1
    #
    i2 = i1 + 1
    while i2 < len(nums) - 1:
      if i2 >= 2 and i2 != i1 + 1:
        while i2 < (len(nums) - 2) and nums[i2] == nums[i2 - 1]:
          i2 += 1
      #
      l, r = i2 + 1, len(nums) - 1
      while l < r:
        fSum = nums[i1] + nums[i2] + nums[l] + nums[r]
        if fSum == target:
          result.append([nums[i1], nums[i2], nums[l], nums[r]])
          l += 1
          while l < len(nums) - 1 and nums[l] == nums[l - 1]:
            l += 1
          r -= 1
          continue
        elif fSum < target:
          l += 1
          while l < len(nums) - 1 and nums[l] == nums[l - 1]:
            l += 1
          continue
        else: # fSum > target
          r -= 1
        #
      i2 += 1
    i1 += 1

  return result

def findSubstring(s, words):
  result = []
  wordsLen = len(words)
  strLen = len(words[0])
  words = Counter(words)

  for offSet in range(strLen):
    for i in range(offSet, len(s), strLen):
      t = dict()
      a = i
      while i < a + strLen*wordsLen:
        subStr = s[i: i + strLen]
        if subStr in words:
          if (subStr in t):
            if (t[subStr] < words[subStr]):
              t[subStr] += 1
            else:
              break
          else:
            t[subStr] = 1
        else:
          break
        #
        i += strLen
      else:
        result.append(a)
  
  # for offSet in range(strLen):
  #   for i in range(offSet, len(s), strLen):
  #     t = dict()
  #     a = i
  #     while i < a + strLen*wordsLen:
  #       subStr = s[i: i + strLen]
  #       if subStr in words:
  #         if (subStr in t):
  #           if (t[subStr] < words[subStr]):
  #             t[subStr] += 1
  #           else:
  #             break
  #         else:
  #           t[subStr] = 1
  #       else:
  #         break
  #       #
  #       i += strLen
  #     else:
  #       result.append(a)

  return result

  # result = []
  # #
  # allPermutation = set()
  # for case in permutations(range(len(words)), len(words)):
  #   concatenated = ''
  #   for w in case:
  #     concatenated += words[w]
  #   allPermutation.add(concatenated)
  # #
  # print(list(allPermutation))
  # for i in range(len(s)):
  #   if s[i: i + len(words[0])] in words:
  #     if s[i: i + len(words)*len(words[0])] in allPermutation:
  #       result.append(i)

  # return result

def canReach(arr, start):

  def _run_(arr = arr, a = set(range(len(arr))), iDex = start):
    if iDex < 0 or iDex >= len(arr): return False
    if arr[iDex] == 0: return True
    if iDex not in a:
      return False
    # have not traveled to index iDex yet
    a.remove(iDex)
    return _run_(iDex = (iDex + arr[iDex])) or _run_(iDex = (iDex - arr[iDex]))
  
  return _run_()
  
def getIntersectionNode(headA, headB):
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  NumberOfListA = 0
  NumberOfListB = 0
  tailNodeA = headA
  tailNodeB = headB

  while tailNodeA.next and tailNodeB.next:
    NumberOfListA += 1
    NumberOfListB += 1
    tailNodeA = tailNodeA.next
    tailNodeB = tailNodeB.next
  while tailNodeA.next:
    tailNodeA = tailNodeA.next
  while tailNodeB.next:
    tailNodeB = tailNodeB.next
  
  #
  nodesFromtail = 0
  while tailNodeA and tailNodeB:
    nodesFromtail += 1
    if tailNodeA != tailNodeB:
      return tailNodeB

def romanToInt(s):
  d = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

  result = 0
  for i in range(len(s) - 1):
    if d[s[i]] < d[s[i + 1]]:
      result -= d[s[i]]
    else:
      result += d[s[i]]
  result += d[s[-1]]
  return result

#

def binaryTreePaths(root):
  # class TreeNode:
  #     def __init__(self, val=0, left=None, right=None):
  #         self.val = val
  #         self.left = left
  #         self.right = right
  result = []

  def _runPath_(path = '', node = root):
    if not node:
      result.append(path)
      return
    #
    if node == root:
      path += node.val
    else:
      path += f'->{node.val}'
    #
    if node.left:
      _runPath_(path, node.left)
    if node.right:
      _runPath_(path, node.right)
  
  _runPath_()
  return result
  
def findWords1(board, words):
  def _run_(board, word, c, r, wordIndex = 0, preMoves = None):
    if wordIndex == len(word):
      return True
    #
    if c == len(board[0]) or c < 0 or r == len(board) or r < 0:
      return False
    #
    if preMoves is None: preMoves = set()
    if (r, c) not in preMoves:
      if board[r][c] != word[wordIndex]:
        return False
      else:
        preMoves.add((r, c))
        if _run_(board, word, c + 1, r, wordIndex + 1, preMoves):
          return True
        preMoves.add((r, c))
        if _run_(board, word, c, r + 1, wordIndex + 1, preMoves):
          return True
        preMoves.add((r, c))
        if _run_(board, word, c - 1, r, wordIndex + 1, preMoves):
          return True
        preMoves.add((r, c))
        if _run_(board, word, c, r - 1, wordIndex + 1, preMoves):
          return True
        else:
          preMoves.remove((r, c))
          return False
    else:
      return False
    
  words = set(words)
  result = []
  found = set()
    
  for row in range(len(board)):
    for col in range(len(board)):
      for word in words:
        if board[row][col] == word[0]:
          if _run_(board, word, col, row): # if word is in board
            found.add(word)
            result.append(word)
      #
      words -= found
      found.clear()

  return result

class Trie:
  def __init__(self, words = None):
    self.root = []
    self.wordCount = 0
    #
    if words is not None:
      for word in words:
        self.add(word)

  class Node:
    def __init__(self, val):
      self.val = val
      self.nexts = []
      self.isWord = False

  def add(self, inputWord):
    self.wordCount += 1
    #
    def _add_(word, node, i):
      if len(word) == 1:
        t = self.Node(word)
        t.isWord = word
        node.nexts.append(t)
        print('flag1')
        return
      #
      if i == len(word):
        node.isWord = word
      #
      if i < len(word):
        print('flag2')
        for n in node.nexts:
          if n.val == word[i]:
            _add_(word, n, i + 1)
            return
        # if no child found
        t = self.Node(inputWord[i])
        #
        if i == len(inputWord) - 1:
          t.isWord = inputWord
        #
        node.nexts.append(t)
        _add_(word, t, i + 1)

    for node in self.root:
      if node.val == inputWord[0]:
        _add_(inputWord, node, 1)
        break
    else:
      t = self.Node(inputWord[0])
      if len(inputWord) == 1:
        t.isWord = inputWord
        self.root.append(t)
        return
      _add_(inputWord, t, 1)

  def delWord(self, word):
    self.wordCount -= 1
    #
    def _run_(word, i, branch, nodeDel, cNode):
      if i == len(word): # end of word
        if len(cNode.nexts) == 0:
          branch.remove(nodeDel)
        else:
          cNode.isWord = False
      else:
        if len(cNode.nexts) >= 2 or cNode.isWord:
          branch = cNode.nexts
          for n in cNode.nexts:
            if n.val == word[i]:
              _run_(word, i + 1, branch, n, n)
              break
        else: # no branching
          _run_(word, i + 1, branch, nodeDel, cNode.nexts[0])

    for node in self.root:
      if node.val == word[0]:
        _run_(word, 1, self.root, node, node)
        break
    else:
      print(f"There is no word '{word}'")

  def allWords(self):
    result = []
    #
    def _run_(node):
      if node.isWord:
        result.append(node.isWord)
      #
      for n in node.nexts:
        _run_(n)
    #
    for node in self.root:
      _run_(node)
    return result
  
  def _delRandWord_(self): # for deletion debugging
    words = self.allWords()
    print(f"{'-'*20}\n", self.allWords())

    for _ in range(len(words)):
      t = words[randint(0, len(words) - 1)]
      self.delWord(t)
      print(f"deleted '{t}' -> {self.allWords()}")
      words.remove(t)
    print(f"{'-'*20}")

# trie = Trie(["a","aa"])
# print(trie.allWords())

def findWords(board, words):
  class Trie:
    def __init__(self, words=None):
      self.root = []
      self.wordCount = 0
      if words is not None:
        for word in words:
          self.add(word)

    class Node:
      def __init__(self, val):
        self.val = val
        self.nexts = []
        self.endOfWord = None

    def add(self, inputWord):
      self.wordCount += 1
      if not inputWord:
        return
      #
      i = 0
      for node in self.root:
        if node.val == inputWord[0]:
          cur = node
          break
      else:
        cur = self.Node(inputWord[0])
        self.root.append(cur)
      #
      if len(inputWord) == 1:
        cur.endOfWord = inputWord
        return
      #
      while i < len(inputWord):
        for n in cur.nexts:
          if n.val == inputWord[i]:
            cur = n
            break
        else:
          new = self.Node(inputWord[i])
          cur.nexts.append(new)
          cur = new
        #
        i += 1
      cur.endOfWord = inputWord

    def allWords(self):
      result = []
      def _run_(node):
        if node.endOfWord:
          result.append(node.endOfWord)
        for n in node.nexts:
          _run_(n)
      for node in self.root:
        _run_(node)
      return result
    
    def _delRandWord_(self): # for deletion debugging
      words = self.allWords()
      print(f"{'-'*20}\n", self.allWords())

      for _ in range(len(words)):
        t = words[randint(0, len(words) - 1)]
        self.delWord(t)
        print(f'deleted {t} -> {self.allWords()}')
        words.remove(t)
      print(f"{'-'*20}")

  result = set()

  def _run_(r, c, trie, cNode, preMoves = None, board = board):
    if c == len(board[0]) or c < 0 or r == len(board) or r < 0:
      return
    #
    if preMoves is None: preMoves = set()
    if (r, c) not in preMoves:
      if board[r][c] != cNode.val:
        return
      else:
        if cNode.endOfWord and cNode.endOfWord not in result:
            result.add(cNode.endOfWord)
        #
        preMoves.add((r, c))
        for n in cNode.nexts:
          if trie.wordCount == 0: break
          #
          _run_(r, c + 1, trie, n, preMoves.copy())
          _run_(r + 1, c, trie, n, preMoves.copy())
          _run_(r, c - 1, trie, n, preMoves.copy())
          _run_(r - 1, c, trie, n, preMoves.copy())
    else:
      return False
        
  # set up trie
  trie = Trie(words)
  print(words)
  print('all words: ', trie.allWords())
  for r in board: print(r)
  #
  
  for row in range(len(board)):
    for col in range(len(board[0])):
      for node in trie.root: # loop 
        if node.val == board[row][col]:
          _run_(row, col, trie, node)
          continue
  
  return list(result)

def countNodes(root):
  def _run_(node = root):
    if not node:
      return -1
    else:
      return 2 + _run_(node.left) + _run_(node.right)
    
  return 1 + _run_()

def invertTree(root):
  # Given the root of a binary tree, invert the tree, and return its root.
  def _run_(node = root):
    if not node:
      return
    else:
      node.left, node.right = node.right, node.left
      _run_(node.left)
      _run_(node.right)
  
  _run_()

def addDigits(num):
  while num > 9:
    s = 0
    while num != 0:
      print(num)
      s += num % 10
      num //= 10
    num = s
  return num

def isUgly(n):
  if n < 0:
    return False
  #
  factor = 2
  while factor <= n:
    if factor > 5: return False
    #
    if n % factor == 0:
      n /= factor
    else:
      factor += 1
  return True

def pathSum(root, targetSum):
  count = [0]

  def _run_(node, targetSum, count, path = None):
    if path is None: path = []
    for i in range(len(path)):
      path[i] += node.val
      if path[i] == targetSum:
        count[1] += 1
    #
    path.append(node.val)
    if node.val == targetSum:
      count[1] += 1
    #
    if node.left:
      _run_(node.left, targetSum, count, path)
    if node.right:
      _run_(node.left, targetSum, count, path)
    #
    path.pop()
    for i in range(len(path)):
      path[i] -= node.val

  if root:
    _run_(root, targetSum, count)
    return count[0]
  else:
    return count[0]
  
def isIsomorphic(s, t):
  # constraint: len(s) == len(t)
  mapping = {}
  mapping2 = {}

  for i in range(len(s)):
    if s[i] in mapping:
      if mapping[s[i]] == t[i]:
        continue
      else:
        return False
    else:
      if t[i] in mapping2:
        return False
      else:
        mapping2[t[i]] = s[i]
        mapping[s[i]] = t[i]
  
  return True

def reverseList(head):

  def _run_(oldNode, newNode):
    if oldNode:
      t = ListNode(oldNode.val)
      t.next = newNode
      oldNode = oldNode.next
      #
      return _run_(oldNode, t)
    else:
      return newNode

  return _run_(head, None)

def findTheDifference(s, t):
  result = 0
  for c in (s + t):
    result ^= c
  return result

def predictTheWinner(nums):

  def _run_(p1Score, p2Score, nums):
    if len(nums) == 0:
      print(p1Score, p2Score)
      return True if p1Score >= p2Score else False
    else:
      return _run_(p1Score + nums[0], p2Score + nums[-1], nums[1:-1]) or _run_(p1Score + nums[-1], p2Score + nums[0], nums[1:-1])

  return _run_(0, 0, nums)

def isSubsequence(s, t):
  sIndex = 0
  tIndex = 0

  while sIndex < len(s) and tIndex < len(t):
    if s[sIndex] == t[tIndex]:
      print(s[sIndex], t[tIndex])
      sIndex += 1
    tIndex += 1
  #
  return True if sIndex == len(s) else False

def isBalanced(root):

  def _run_(node):
    if not node:
      return 0
    #
    a, b = 0, 0
    if node.left:
      a = 1 + _run_(node.left)
    if node.right:
      b = 1 + _run_(node.right)
    return max(a, b)
  
  def _check_(node):
    if not node:
      return True
    else:
      if abs(_run_(node.left) - _run_(node.right)) <= 1:
        return _check_(node.left) or _check_(node.right)
      else:
        return False
      
  return _check_(root)

def getIntersectionNode(headA, headB):
  nodeA = set()
  nodeB = set()

  while headA and headB:
    if headA not in nodeB:
      nodeA.add(headA)
    else:
      return headA
    #
    if headB not in nodeA:
      nodeB.add(headB)
    else:
      return headB
    #
    headA = headA.next
    headB = headB.next

  while headA:
    if headA not in nodeB:
      headA = headA.next
    else:
      return headA
    
  while headB:
    if headB not in nodeA:
      headB = headB.next
    else:
      return headB
    
def rotate(nums, k):
  """
  Do not return anything, modify nums in-place instead.
  """
  k = k % len(nums)
  partA = nums[:len(nums) - k]
  partB = nums[len(nums) - k:]
  for i in range(len(partB)):
    nums[i] = partB[i]
  for i in range(len(partA)):
    nums[len(partB) + i] = partA[i]
  return nums

def combinationSum2(candidates, target):
  #   Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

  # Each number in candidates may only be used once in the combination.
  candidates.sort()
  print(candidates, target, '\n------------')
  result = []

  def _run_(_index, subCandidates, _sum = 0):
    if _sum == target:
      result.append(subCandidates)
      return False
    if _sum > target:
      return False  
    #
    for i in range(_index + 1, len(candidates)):
      subCandidates.append(candidates[i])
      if not _run_(i, subCandidates, _sum + candidates[i]):
        subCandidates.pop()
        break
      subCandidates.pop()
    else:
      return True
    
    return False
  
  for i in range(len(candidates)):
    _run_(i, [candidates[i]], candidates[i])

  return result

def spiralMatrix(self, m: int, n: int, head: Optional[ListNode]) -> List[List[int]]:
  top = 0
  right = n - 1
  bottom = m - 1
  left = 0

  result = [[-1 for _ in range(n)] for _ in range(m)]
  while top <= bottom and left <= right:
    # move right
    for i in range(left, right + 1):
      if head:
        result[top][i] = head.val
        head = head.next
      else:
        break
    top += 1

    # move down
    for i in range(top, bottom + 1):
      if head:
        result[i][right] = head.val
        head = head.next
      else:
        break
    right -= 1

    # move left
    for i in range(right, left - 1, -1):
      if head:
        result[bottom][i] = head.val
        head = head.next
      else:
        break
    bottom -= 1

    # move up
    for i in range(bottom, top - 1, -1):
      if head:
        result[i][left] = head.val
        head = head.next
      else:
        break
    left += 1

  return result

def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, val=0, next=None):
  #         self.val = val
  #         self.next = next
  dummy = ListNode(next = head)

def sortedListToBST(head: Optional[ListNode]) -> Optional[TreeNode]:
  nodes = []
  while head:
    nodes.append(head.val)
    head = head.next

  if len(nodes) == 0: return None
  def construct(l, r):
    if l > r:
      return None
    else:
      mid = (l + r) // 2
      temp = ListNode(nodes[mid])
      temp.left = construct(l, mid - 1)
      temp.right = construct(mid + 1, r)
      return temp
  
  mid = len(nodes) // 2
  root = ListNode(nodes[mid])
  root.left = construct(0, mid - 1)
  root.right = construct(mid + 1, len(nodes) - 1)

  return root

def permute(nums: List[int]) -> List[List[int]]:
  result = []

  def _run_(p):
    if len(p) == len(nums):
      result.append(p)
      return
    #
    for i in nums:
      if i not in p:
        p.append(i)
        _run_(p)

  _run_([])
  return result

def maxSubArray(self, nums: List[int]) -> int:
  _max = int('-inf')
  tMax = 0

  for num in nums:
    tMax += num

    if tMax < num:
      tMax = num

    if tMax > _max:
      _max = tMax

  return _max

def rotate(matrix) -> None:
  """
  Do not return anything, modify matrix in-place instead.
  """
  tMatrix = [[matrix[b][a] for b in range(len(matrix) - 1, -1, -1)] for a in range(len(matrix))]
  for i in range(len(matrix)):
    for j in range(len(matrix)):
      matrix[i][j] = tMatrix[i][j]

def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
  _count = 0
  t = head
  while t:
    t = t.next
    _count += 1
  #
  if _count <= 1 or k == 0: return head
  k = k % _count
  if k == 0: return head
  #
  newTail = head
  for _ in range(_count - 1 - k):
    newTail = newTail.next
  newHead = newTail.next
  #
  newTail.next = None
  oldTail = newHead
  while oldTail.next:
    oldTail = oldTail.next
  oldTail.next = head
  
  return newHead

def uniquePaths(m: int, n: int) -> int:
  matrix = [[0 for _ in range(m)] for _ in range(n)]
  for i in range(n):
    for j in range(m):
      if i == 0 or j == 0:
        matrix[i][j] = 1
      else:
        matrix[i][j] = matrix[i - 1][j] + matrix[i][j - 1]
  return matrix[n - 1][m - 1]

def uniquePathsWithObstacles(obstacleGrid: List[List[int]]) -> int:
  for i in range(len(obstacleGrid)):
    for j in range(len(obstacleGrid[0])):
      if obstacleGrid[i][j] == 1:
        obstacleGrid[i][j] = 'o'
  #
  for row in obstacleGrid:
    print(row)
  #
  if obstacleGrid[0][0] != 'o' and obstacleGrid[-1][-1] != 'o':
    obstacleGrid[0][0] = 1
  else:
    return 0
  #
  for i in range(len(obstacleGrid)):
    for j in range(len(obstacleGrid[0])):
      if obstacleGrid[i][j] != 'o':
        up = obstacleGrid[i - 1][j] if i - 1 >= 0 else 0
        down = obstacleGrid[i][j - 1] if j - 1 >= 0 else 0
        print(obstacleGrid[i][j], up, down)
        obstacleGrid[i][j] += (up if up != 'o' else 0) + (down if down != 'o' else 0)
  #
  for row in obstacleGrid:
    print(row)
  #
  return obstacleGrid[-1][-1]

def divide(dividend: int, divisor: int) -> int:
  if dividend == -2147483648 and divisor == -1:
    return 2147483647
  def run(dividend, divisor, quotient):
    shiftAmount = 0
    while divisor << shiftAmount <= dividend:
      shiftAmount += 1
    #
    shiftAmount -= 1
    if shiftAmount == -1:
      return quotient
    elif shiftAmount == 0:
      return quotient + 1
    else:
      dividend -= divisor << shiftAmount
      quotient += 2 << shiftAmount - 1
      return run(dividend, divisor, quotient)

  result = run(abs(dividend), abs(divisor), 0)
  if (dividend >= 0 and divisor >= 0) or (dividend < 0 and divisor < 0):
    return result
  else:
    return -result

def partition(head: Optional[ListNode], x: int) -> Optional[ListNode]:
  # class ListNode:
  #     def __init__(self, val=0, next=None):
  #         self.val = val
  #         self.next = next
  # y would be the value that greater than or equal to x
  y = x
  nodesLessthanX = []
  nodesGreaterThanX = []

  t = head
  foundY = False
  while t:
    if not foundY and t.val > y:
      y = t.val
      foundY = True
    t = t.next
  #
  t = head
  while t:
    if t.val < x:
      nodesLessthanX.append(t.val)
    else:
      nodesGreaterThanX.append(t.val)
    t = t.next

  newHead = ListNode()
  t = newHead
  for i in (nodesLessthanX + nodesGreaterThanX):
    t.next = ListNode(i)
  return newHead.next

def merge(intervals: List[List[int]]) -> List[List[int]]:
  f = [_ for i in sorted(intervals) for _ in i]
  i = 1

  while i < len(f) - 2:
    if f[i] >= f[i + 1]:
      if f[i] >= f[i + 2]:
        del f[i + 1: i + 3]
      else: del f[i], f[i]
    else: i += 2

  return [[f[i], f[i + 1]] for i in range(0, len(f), 2)]

def subsets(nums: List[int]) -> List[List[int]]:

  result = []
  def _run_(subSet):
    subSet.sort()
    if subSet not in result:
      result.append(subSet.copy())
    for i in nums:
      if i not in subSet:
        subSet.append(i)
        _run_(subSet)
        subSet.pop()

  _run_([])
  # result = [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [3]]
  sorted(result, key = lambda x: len(x))
  return result

def sumOfLeftLeaves(root: Optional[TreeNode]) -> int:
  result = []
  def _run(node, isLeft, result):
    if node.left is None and node.right is None:
      if isLeft:
        result[0] += node.val
    else:
      if node.left: 
        _run(node.left, True)
      if node.right:
        _run(node.right, False)

  _run(root, False)
  return result[0]

def longestPalindrome(s: str) -> int:
  existOddValue = False
  hashTable = {}
  for i in s: # s: string
    if i in hashTable:
      hashTable[i] += 1
    else:
      hashTable[i] = 1
  #
  if existOddValue:
    result = 1
  else:
    result = 0
  #
  for s in hashTable.values():
    result += (s - s%2)
    if s % 2 == 1:
      existOddValue = True
  return result + (1 if existOddValue else 0)

def removeElements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
  dum = ListNode(-1, head)
  t = dum
  while t.next:
    if t.next.val == val:
      t.next = t.next.next
    t = t.next
  return dum.next

def setZeroes(matrix: List[List[int]]) -> None:
  s = []
  for i in range(len(matrix)):
    for j in range(len(matrix[0])):
      if matrix[i][j] == 0: s.append([i,j])
  #
  for p in s:
    for i in range(len(matrix)):
      matrix[i][p[1]] = 0
    for i in range(len(matrix[0])):
      matrix[p[0]][i] = 0

def numIslands(grid: List[List[str]]) -> int:
    def run(post, grid, p):
      if grid[p[0]][p[1]] == '1':
        post.remove(p)
        if (p[0], p[1] - 1) in post:
          run(post, grid, (p[0], p[1] - 1))
        if (p[0], p[1] + 1) in post:
          run(post, grid, (p[0], p[1] + 1))
        if (p[0] - 1, p[1]) in post:
          run(post, grid, (p[0] - 1, p[1]))
        if (p[0] + 1, p[1]) in post:
          run(post, grid, (p[0] + 1, p[1]))

    islandCount = 0
    positions = {(i, j) for i in range(len(grid)) for j in range(len(grid[0]))}
    postSet = {(i, j) for i in range(len(grid)) for j in range(len(grid[0]))}
    for i in positions:
      if i in postSet and grid[i[0]][i[1]] == '1':
        run(postSet, grid, i)
        islandCount += 1
    
    return islandCount

def missingNumber(nums: List[int]) -> int:
  return len(nums)*(len(nums) + 1) / 2 - sum(nums)


def findWords(board: List[List[str]], words: List[str]) -> List[str]: 
  class Trie:
    def __init__(self, words=None):
      self.root = []
      self.wordCount = 0
      if words is not None:
        for word in words:
          self.add(word)

    class Node:
      def __init__(self, val):
        self.val = val
        self.nexts = []
        self.endOfWord = None

    def add(self, inputWord):
      self.wordCount += 1
      #
      for node in self.root:
        if node.val == inputWord[0]:
          cur = node
          break
      else:
        cur = self.Node(inputWord[0])
        self.root.append(cur)
      #
      if len(inputWord) == 1:
        cur.endOfWord = inputWord
        return
      # For longer words, walk/create child nodes
      for ch in inputWord[1:]:
        for n in cur.nexts:
          if n.val == ch:
            cur = n
            break
        else:
          new = self.Node(ch)
          cur.nexts.append(new)
          cur = new
      cur.endOfWord = inputWord

    def allWords(self):
      result = []
      def _run_(node):
        if node.endOfWord:
          result.append(node.endOfWord)
        for n in node.nexts:
          _run_(n)
      for node in self.root:
        _run_(node)
      return result

  result = set()

  def _run_(r, c, trie, cNode, preMoves = None, board = board):
    if c == len(board[0]) or c < 0 or r == len(board) or r < 0:
      return
    #
    if preMoves is None: preMoves = set()
    if (r, c) not in preMoves:
      if board[r][c] != cNode.val:
        return
      else:
        if cNode.endOfWord and cNode.endOfWord not in result:
            result.add(cNode.endOfWord)
        #
        preMoves.add((r, c))
        for n in cNode.nexts:
          if trie.wordCount == 0: break
          #
          _run_(r, c + 1, trie, n, preMoves.copy())
          _run_(r + 1, c, trie, n, preMoves.copy())
          _run_(r, c - 1, trie, n, preMoves.copy())
          _run_(r - 1, c, trie, n, preMoves.copy())
    else:
      return False
        
  # set up trie
  trie = Trie(words)
  #
  
  for row in range(len(board)):
    for col in range(len(board[0])):
      for node in trie.root: # loop 
        if node.val == board[row][col]:
          _run_(row, col, trie, node)
          continue
  
  return list(result)

def summaryRanges(nums: List[int]) -> List[str]:
  result = []

  head = 0
  while head < len(nums):
    tail = head + 1
    while tail < len(nums):
      if nums[tail] == nums[tail - 1] + 1:
        tail += 1
      else:
        break
    #
    print(head, tail)
    if head == tail - 1:
      result.append(f"{nums[head]}")  
    else:
      result.append(f"{nums[head]}->{nums[tail - 1]}")
    head = tail
  #
  return result

def areaOfMaxDiagonal(dimensions: List[List[int]]) -> int:
  maxArea = 1
  maxTempDia = 1
  for rectangle in dimensions:
    t = rectangle[0]*rectangle[0] + rectangle[1]*rectangle[1]
    print(math.sqrt(t), rectangle[0]*rectangle[1])
    if t == maxTempDia:
      t = rectangle[0]*rectangle[1]
      if t > maxArea:
        maxArea = t
    elif t > maxTempDia:
      maxTempDia = t
      maxArea = rectangle[0]*rectangle[1]
  
  return maxArea

def maximumSetSize(nums1: List[int], nums2: List[int]) -> int:
  nums1DelCount = nums2DelCount = int(len(nums1))/2
  countNums1 = Counter(nums1)
  countNums2 = Counter(nums2)
  #
  for i in countNums1:
    if countNums1[i] > 1:
      t = countNums1[i] - 1
      if t >= nums1DelCount:
        countNums1[i] -= nums1DelCount
        nums1DelCount = 0
        break
      else:
        nums1DelCount -= countNums2[i]
        countNums1[i] = 1
  for i in countNums2:
    if countNums2[i] > 1:
      t = countNums2[i] - 1
      if t > nums2DelCount:
        countNums2[i] -= nums2DelCount
        nums2DelCount = 0
        break
      else:
        nums2DelCount -= countNums2[i]
        countNums2[i] = 1
  #
  if nums1DelCount <= 0 and nums2DelCount <= 0:
    return len(Counter(nums1 + nums2))
  #
  for i in countNums1.copy():
    if i in countNums2:
      if countNums1[i] > nums1DelCount:
        nums1DelCount = 0
        break
      else:
        nums1DelCount -= countNums1[i]
        del countNums1[i]
    else:
      continue
  
  for i in countNums2.copy():
    if i in countNums1:
      if countNums2[i] > nums2DelCount:
        nums2DelCount = 0
        break
      else:
        nums2DelCount -= countNums2[i]
        del countNums2[i]
  
  return int(len(Counter(countNums1 + countNums2)) - nums1DelCount - nums2DelCount)

def minCost(m: int, n: int, waitCost: List[List[int]]) -> int:
  t = waitCost[m - 1][n - 1]
  waitCost[0][0] = 1
  for row in range(m):
    for col in range(n):
      if row == 0 and col == 0: continue
      if row == 0:
        waitCost[row][col] += waitCost[row][col - 1] + (row + 1)*(col + 1)
        continue
      if col == 0:
        waitCost[row][col] += waitCost[row - 1][col] + (row + 1)*(col + 1)
        continue
      #
      waitCost[row][col] += min(waitCost[row][col - 1], waitCost[row - 1][col]) + (row + 1)*(col + 1)
  
  return waitCost[m - 1][n - 1] - t

def splitArray(nums: List[int]) -> int:
  def insertNewPrime(p):
    newPrime = p[-1] + 2
    while True:
      for i in p:
        if i*i > newPrime:
          p.append(newPrime)
          return
        else:
          if newPrime % i != 0:
            continue
          else:
            break
      
      newPrime += 1
          
  primes = [2,3]
  _index = 0

  if len(nums) <= 2:
    sumA = 0
  else:
    sumA = nums[2]
  sumB = 0
  while _index < len(nums):
    if _index == primes[-1]:
      sumA += nums[_index]
      insertNewPrime(primes)
    else:
      if _index != 2:
        sumB += nums[_index]

    _index += 1

  return abs(sumA - sumB)

def longestBalanced(nums: List[int]) -> int:
  result = 0
  temp = []
  for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
      evenCount = 0
      oddCount = 0
      k = collections.Counter(nums[i:j + 1])
      for num in k:
        if num % 2 == 0:
          evenCount += 1
        else:
          oddCount += 1
      
      if evenCount == oddCount:
        t = j - i + 1
        if t > result:
          result = t
          #
          temp = [i, j]
      else:
        continue

  return result, nums[temp[0]:temp[1] + 1]

#print([longestBalanced([15634,46946,17078,30747,11127,8734,34499,40653,69290,58196,59189,56437,62193,32120,27951,13810,18249,56184,13632,62142,22005,53022,7646,20459,69629,61727,58260,68267,23311,25461,59165,43969,68956,67035,29041,49997,43759,67083,10816,55527,16304,2808,7590,37814,40137,3366,71407,40267,62902,45332,55681,48864,64191,13856,25594,50714,41507,45931,15778,37523,30090,19141,60623,25036,34743,8611,60783,41444,2012,16841,61705,64154,9302,64601,5060,54224,54179,58568,65371,42090,24509,51791,8024,22459,22880,40109,71815,13525,36507,23858,63518,16087,54501,9894,21935,10884,70488,59708,7560,49077,33641,40109,22320,34285,16926,65912,15054,71166,34398,53472,67492,57202,36698,26486,14855,62050,53039,26264,29600,17548,64755,36430,22401,37778,30335,25061,64221,42431,34821,8401,34200,55539,4698,8553,26720,36335,61536,67866,9718,61787,33770,19887,69223,37792,38458,68774,60529,69679,5010,45098,10741,42222,48397,40795,46588,28645,43910,25233,48711,42584,6526,52948,68149,20110,60468,48395,50760,50031,52699,11252,66136,62403,46617,21753,46072,61626,63986,25977,1013,48138,67805,14462,14780,35262,48151,61807,44182,43909,59102,15572,46369,35328,20094,69825,69466,17869,38007,69566,40524,43931,14345,59494,26982,2251,10429,26922,7972,7787,57306,40449,11896,50577,66200,66185,27493,63952,60438,69607,40903,24008,53578,33890,49433,30553,60612,26496,49501,25129,21052,9476,593,56888,52830,21418,53550,61195,70284,58698,12824,14110,66236,38414,6480,47018,1788,66511,13175,27445,27467,32970,21245,20161,66692,41282,17232,31938,5208,41678,53878,60186,19244,29640,48313,52298,4587,24579,30078,18651])])


  


def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	try:
		return np.reshape(a, new_shape).tolist()
	except Exception as e:
		return []

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
  if mode == 'column':
    return [sum([matrix[j][i] for j in range(len(matrix))])/len(matrix) for i in range(len(matrix[0]))]
  else:
    return [sum([matrix[i][j] for j in range(len(matrix[0]))])/len(matrix[0]) for i in range(len(matrix))]
  
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
  for row in range(len(matrix)):
    for col in range(len(matrix[0])):
      matrix[row][col] *= scalar

  return matrix