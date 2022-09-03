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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        
        x_max = len(list(newFood))
        y_max = len(newFood[0])

        dist=[]
        for x in range(x_max):
            for y in range(y_max):
                if newFood[x][y]:
                    dist.append(manhattanDistance((x, y), newPos))

        for ghost in newGhostStates:
            if ghost.scaredTimer <= 0:
                if util.manhattanDistance(newPos,successorGameState.getGhostPosition(newGhostStates.index(ghost) + 1)):
                    score -= 150
                else:
                    score -= 50
            else:
                #sinon on peut le manger : tant mieux
                score += 100
    

        if currentGameState.getFood()[newPos[0]][newPos[1]] :
            return score
        else:
            return score - min(dist)


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.minimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]

    def minimaxSearch(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            res = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            res = self.mSearch(gameState, agentIndex, depth,-1)
        else:
            res = self.mSearch(gameState, agentIndex, depth, 1)
        return res

    def mSearch(self, gameState, agentIndex, depth, m):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex < gameState.getNumAgents() - 1:
            next_agent, next_depth = agentIndex + 1, depth
        else:
            next_agent, next_depth = 0, depth - 1
        l = [(m*10000,Directions.STOP)]

        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score = self.minimaxSearch(successor_game_state, next_agent, next_depth)[0]
            l.append((new_score, action))

        if m ==1:
            return min(l)
        else:
            return max(l)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabetaSearch(gameState, agentIndex=0, depth=self.depth, alpha=-10000, beta=10000)[1]

    def alphabetaSearch(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            res = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            res = self.alphabetasubSearch(gameState, agentIndex, depth, alpha, beta,-1)
        else:
            res = self.alphabetasubSearch(gameState, agentIndex, depth, alpha, beta, 1)
        return res

    def alphabetasubSearch(self, gameState, agentIndex, depth, alpha, beta, mode):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex < gameState.getNumAgents() - 1:
            next_agent, next_depth = agentIndex + 1, depth
        else:
            next_agent, next_depth = 0, depth - 1
        l = [(mode*10000,Directions.STOP)]

        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score = self.alphabetaSearch(successor_game_state, next_agent, next_depth, alpha, beta)[0]
            l.append((new_score, action))
            if mode == 1:
                if new_score < alpha:
                    return (new_score, action)
                beta = min(beta, min(l)[0])
            else:
                if new_score > beta:
                    return (new_score, action)
                alpha = max(alpha, max(l)[0])

        if mode == 1:
            return min(l)
        else:
            return max(l)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]

    def expectimaxSearch(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            res = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            res = self.expectimaxsubSearch(gameState, agentIndex, depth,-1)
        else:
            res = self.expectimaxsubSearch(gameState, agentIndex, depth, 0)
        return res

    def expectimaxsubSearch(self, gameState, agentIndex, depth, mode):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex < gameState.getNumAgents() - 1:
            next_agent, next_depth = agentIndex + 1, depth
        else:
            next_agent, next_depth = 0, depth - 1
        l = [(mode*10000,Directions.STOP)]

        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score = self.expectimaxSearch(successor_game_state, next_agent, next_depth)[0]
            l.append((new_score, action))

        if mode == -1:
            return max(l)
        else:
            return (sum([i[0] for i in l])/len(actions),Directions.STOP)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: On reprend juste le ReflexAgent et on pénalise davantage
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()
    
    x_max = len(list(newFood))
    y_max = len(newFood[0])

    dist=[]
    for x in range(x_max):
        for y in range(y_max):
            if newFood[x][y]:
                dist.append(manhattanDistance((x, y), newPos))

    for ghost in newGhostStates:
        if ghost.scaredTimer <= 0:
            if util.manhattanDistance(newPos,currentGameState.getGhostPosition(newGhostStates.index(ghost) + 1)):
                score -= 350
            else:
                score -= 250
        else:
            #sinon on peut le manger : tant mieux
            score += 100


    if currentGameState.getFood()[newPos[0]][newPos[1]] or len(dist) == 0:
        return score
    else:
        return score - min(dist)

# Abbreviation
better = betterEvaluationFunction
