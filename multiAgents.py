# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #print newGhostStates[0].getPosition()
        "*** YOUR CODE HERE ***"
        #upologizw to evaluation sumfwna me tis apostaseis apo to food kai to ghost
        score = successorGameState.getScore()

        ghostDist = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if ghostDist:
            score -= 1.0 / ghostDist

        foodPos = newFood.asList()
        #print foodPos
        foodDist = []
        #print foodDist
        #print len(foodPos)
        for f in range(len(foodPos)):
            #print foodPos[f]
            foodDist.append(manhattanDistance(newPos, foodPos[f]))
            #print foodDist
        if len(foodDist):
            minDist = min(foodDist)
            score += 1.0 / (1.0 + minDist)

        return score
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def max_value(self, gameState, agentIndex, depth):
        depth+=1
        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
        #depth+=1
        actions = gameState.getLegalActions(agentIndex)
        v = -float('Inf')

        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            v = max(v, self.min_value(successor, 1, depth))
        #depth+=1
        return v

    def min_value(self, gameState, agentIndex, depth):
        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        v = float('Inf')
        #depth+=1
        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            #elegxw an eimai ston eleutaio agent
            if agentIndex == gameState.getNumAgents()-1:
                last_agent = True
            else:
                last_agent = False
            if last_agent:
                #depth+=1
                #an eimai ston teleutaio kalw ton pacman
                v = min(v, self.max_value(successor, 0, depth))
            else:
                #alliws ksanakalw th min_value gia ton epomeno agent
                v = min(v, self.min_value(successor, agentIndex+1, depth))

        return v

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        action = "None"
        values = []
        agentIndex = self.index
        #kanw oti kai sth max_value gia depth=0
        pacmanActions = gameState.getLegalActions(agentIndex)
        for a in pacmanActions:
            successor = gameState.generateSuccessor(agentIndex, a)
            #apothikeuw lista apo tuples gia na epistrepsw to action sumfwna me to value tou
            values.append((self.min_value(successor, 1, depth), a))
        #maxV = (maxEval, action)
        maxV = max(values)

        return maxV[1]

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self, gameState, agentIndex, depth, a, b):
        depth+=1
        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        v = -float('Inf')

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.min_value(successor, 1, depth, a, b))
            if v > b:
                return v
            a = max(a, v)

        return v

    def min_value(self, gameState, agentIndex, depth, a, b):
        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        v = float('Inf')

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            #elegxw an eimai ston eleutaio agent
            if agentIndex == gameState.getNumAgents()-1:
                last_agent = True
            else:
                last_agent = False

            if last_agent:
                v = min(v, self.max_value(successor, 0, depth, a, b))
            else:
                v = min(v, self.min_value(successor, agentIndex+1, depth, a, b))

            if v < a:
                return v
            b = min(b, v)

        return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        action = "None"
        a = -float('Inf')
        b = float('Inf')
        values = []
        agentIndex = self.index
        pacmanActions = gameState.getLegalActions(agentIndex)
        #kanw oti kai sth max_value gia depth=0
        for ac in pacmanActions:
            successor = gameState.generateSuccessor(agentIndex, ac)
            #apothikeuw lista apo tuples gia na epistrepsw to action sumfwna me to value tou
            values.append((self.min_value(successor, 1, depth, a, b), ac))
            #maxV = (maxEval, action)
            maxV = max(values)
            if maxV[0] > b:
                return maxV[1]
            a = max(a, maxV[0])

        return maxV[1]

        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_value(self, gameState, agentIndex, depth):
        depth+=1
        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        v = -float('Inf')

        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            v = max(v, self.exp_value(successor, 1, depth))

        return v

    def exp_value(self, gameState, agentIndex, depth):
        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        v = 0.0

        for a in actions:
            successor = gameState.generateSuccessor(agentIndex, a)
            #elegxw an eimai ston eleutaio agent
            if agentIndex == gameState.getNumAgents()-1:
                last_agent = True
            else:
                last_agent = False
            if last_agent:
                #ta kanw float
                v += 1.0*self.max_value(successor, 0, depth)
            else:
                v += 1.0*self.exp_value(successor, agentIndex+1, depth)

            chance = v / (1.0*len(gameState.getLegalActions(agentIndex)))

        return chance

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        depth = 0
        action = "None"
        values = []
        agentIndex = self.index
        pacmanActions = gameState.getLegalActions(agentIndex)
        #kanw oti kai sth max_value gia depth=0
        for a in pacmanActions:
            successor = gameState.generateSuccessor(agentIndex, a)
            #apothikeuw lista apo tuples gia na epistrepsw to action sumfwna me to value tou
            values.append((self.exp_value(successor, 1, depth), a))
        #maxV = (maxEval, action)
        maxV = max(values)
        #print maxV[1]

        return maxV[1]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #kanw to idio me to 1o erwthma aplws xrhsimopoiw to current state mono (den einai apoluta swsto)
    Position = currentGameState.getPacmanPosition()
    Foodlist = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    #ScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()

    foodPos = Foodlist.asList()
    foodDist = []
    for f in range(len(foodPos)):
        foodDist.append(manhattanDistance(Position, foodPos[f]))
    if len(foodDist):
        minDist = min(foodDist)
        score += 1.0 / (1.0 + minDist)

    ghostPos = GhostStates[0].getPosition()
    ghostDist = manhattanDistance(Position, ghostPos)

    if ghostDist:
        score -= 1.0 / (1.0 + ghostDist)

    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
