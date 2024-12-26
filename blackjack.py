import numpy as np
from scipy import optimize
import copy
from gekko import GEKKO

class Blackjack:
    # this class is a simple implementation of the game of blackjack
    # the game is played between a dealer and a player
    # the player can draw cards until they decide to stop or they bust
    # the dealer can draw cards until their score is greater than the player's score or they bust
    # the player wins if they have a higher score than the dealer or the dealer busts
    # the player loses if they bust or the dealer has a higher score
    # -----------------------------------------
    # b: the maximum score that can be achieved
    # t: the total number of cards in the deck
    # scores: the scores of the cards
    # a: the alternative score of the ace
    # q_init: the initial policy of the player
    # -----------------------------------------
    def __init__(self, b=21, t=np.array([4]*9 + [16], dtype=np.uint8), scores=np.arange(1,11), a=11, q_init=0.5):
        self.b = b
        self.t = t
        self.scores = scores
        self.a = a
        self.q_init = q_init

        self.Nclass = len(t)
        self.Ncards = np.sum(t)

        self.init_Adiff()
        self.init_state_dict()


    def init_Adiff(self): 
        # initialize the difference between the alternative score of the ace and the score of the ace
        self.Adiff = self.a - self.t[0]

    def num_cards(self, u):
        # return the number of cards in the hand
        return np.sum(u)
    
    def score_cards(self, u):
        # return the score of the hand
        score0 = np.sum(u * self.scores)
        score1 = self.Adiff * max(min([np.floor((self.b-score0)/self.Adiff), u[0]]), 0) # the score of the ace
        return score0 + score1
    
    def score_bust(self, score_u):
        # return whether the score is greater than the maximum score
        return score_u > self.b
    
    def prob_draw(self, h, d):
        # return the probability of drawing a card
        r = self.t - h - d
        r_total = np.sum(r)
        if r_total == 0:
            return np.zeros_like(r)
        return r/r_total
    
    def plus_one(self, u, i):
        # add one card i to the hand
        out = copy.deepcopy(u)
        out[i] += 1
        return out

    def draw_results(self, u, p):
        # return the possible hands after drawing a card
        return [self.plus_one(u, i) for i, p_i in enumerate(p) if p_i > 0]

    def def_win(self, h, d):
        # return the winner of the game when the player has hand h and the dealer has hand d
        score_h = self.score_cards(h)
        score_d = self.score_cards(d)
        if self.score_bust(score_h):
            return 0
        if self.score_bust(score_d):
            return 1
        return (score_h > score_d) * 1
    
    def get_dict_key(self, h, d, c):
        # return the key of the state dictionary
        return np.array2string(h) + np.array2string(d) + str(c)

    def value_cards(self, h, d, c, q):
        # return the value of the game when the player has hand h, the dealer has hand d
        # c is the current player, 0 for the player and 1 for the dealer
        # q is the policy of the player; if q(h,d,c) = 0, the player stops drawing cards and the dealer starts drawing cards; otherwise, the player draws another card
        dict_key = self.get_dict_key(h, d, c)
        
        if dict_key in self.state_values.keys():
            return self.state_values[dict_key]
        
        state_dict = self.state_dict[dict_key]

        if state_dict['num_residual_cards'] == 0:
            self.state_values[dict_key] = state_dict['win']
            return self.state_values[dict_key]

        if state_dict['num_cards_d'] < 1:
            self.state_values[dict_key] = np.sum([self.value_cards(h, dp, c, q) * p_i for dp, p_i in zip(state_dict['d_plus'], state_dict['p'])])
            return self.state_values[dict_key]

        if state_dict['num_cards_h'] < 2:
            self.state_values[dict_key] = np.sum([self.value_cards(hp, d, c, q) * p_i for hp, p_i in zip(state_dict['h_plus'], state_dict['p'])])
            return self.state_values[dict_key]
        
        if state_dict['score_bust_h']:
            self.state_values[dict_key] = 0
            return 0
        if state_dict['score_bust_d']:
            self.state_values[dict_key] = 1
            return 1
        
        if c == 0:
            rate = q(h,d,c)
            state_value_0 = self.value_cards(h, d, 1, q)
            if state_dict['num_residual_cards'] == 0:
                state_value_1 = self.state_dict[dict_key]['win']
            elif state_dict['num_residual_cards'] == 1:
                state_value_1 = state_value_0
            else:
                state_value_1 = np.sum([self.value_cards(hp, d, 0, q) * p_i for hp, p_i in zip(state_dict['h_plus'], state_dict['p'])])
            self.state_values[dict_key] = rate * state_value_0 + (1-rate) * state_value_1
        else:
            state_win = self.state_dict[dict_key]['win']
            if state_dict['num_residual_cards'] == 0:
                state_value = state_win
            elif state_win == 0:
                state_value = state_win
            else:
                state_value = np.sum([self.value_cards(h, dp, 1, q) * p_i for dp, p_i in zip(state_dict['d_plus'], state_dict['p'])])
            self.state_values[dict_key] = state_value
        return self.state_values[dict_key]
        
        '''
        if q(h,d,c) == 0:
            if c == 0:
                self.state_values[dict_key] = self.value_cards(h, d, 1, q)
                return self.state_values[dict_key]
            else:
                self.state_values[dict_key] = self.def_win(h, d)
                return self.state_values[dict_key]
        else:
            if c == 0:
                self.state_values[dict_key] = np.sum([self.value_cards(hp, d, 0, q) * p_i for hp, p_i in zip(state_dict['h_plus'], state_dict['p'])])
                return self.state_values[dict_key]
            else:
                self.state_values[dict_key] = np.sum([self.value_cards(h, dp, 1, q) * p_i for dp, p_i in zip(state_dict['d_plus'], state_dict['p'])])
                return self.state_values[dict_key]
        '''
            
    def evaluate(self, q=None):
        # return the value of the game when the player follows the policy q
        if q is None:
            q = self.get_variable
        self.state_values = {}
        return self.value_cards(np.zeros_like(self.t), np.zeros_like(self.t), 0, q)
            
    def init_state_dict(self):
        # initialize the state dictionary
        self.state_dict = {}
        self.variables = []
        self.variables_idx = {}
        self.init_hdc(np.zeros_like(self.t), np.zeros_like(self.t), 0)
        self.variables = np.array(self.variables)
        self.variables_init = copy.deepcopy(self.variables)
        self.variables_lb = np.zeros_like(self.variables)
        self.variables_ub = np.ones_like(self.variables)

    def init_hdc(self, h, d, c):
        # initialize the state dictionary for the player with hand h, the dealer with hand d, and the current player c
        dict_key = self.get_dict_key(h, d, c)
        if dict_key in self.state_dict.keys():
            return
        
        state_dict = {'p': self.prob_draw(h, d)}
        state_dict['h'] = h
        state_dict['d'] = d
        state_dict['c'] = c
        state_dict['h_plus'] = self.draw_results(h, state_dict['p'])
        state_dict['d_plus'] = self.draw_results(d, state_dict['p'])
        state_dict['score_h'] = self.score_cards(h)
        state_dict['score_d'] = self.score_cards(d)
        state_dict['score_bust_h'] = self.score_bust(state_dict['score_h'])
        state_dict['score_bust_d'] = self.score_bust(state_dict['score_d'])
        state_dict['win'] = self.def_win(h, d)
        state_dict['num_cards_h'] = self.num_cards(h)
        state_dict['num_cards_d'] = self.num_cards(d)
        state_dict['num_residual_cards'] = self.Ncards - state_dict['num_cards_h'] - state_dict['num_cards_d']
        
        self.state_dict[dict_key] = state_dict

        if state_dict['num_cards_d'] < 1:
            for dp in state_dict['d_plus']:
                self.init_hdc(h, dp, c)
            return 

        if state_dict['num_cards_h'] < 2:
            for hp in state_dict['h_plus']:
                self.init_hdc(hp, d, c)
            return 
        
        if state_dict['score_bust_h']:
            return 
        if state_dict['score_bust_d']:
            return 
        
        
        if c == 0:
            self.variables_idx[dict_key] = len(self.variables)
            self.variables.append(self.q_init)
            if state_dict['num_residual_cards'] > 1:
                for hp in state_dict['h_plus']:
                    self.init_hdc(hp, d, 0)
            self.init_hdc(h, d, 1)
            return
        else:
            if state_dict['win'] == 1:
                for dp in state_dict['d_plus']:
                    self.init_hdc(h, dp, 1)
            return
        
    def get_variable(self, h, d, c):
        # return the policy of the player when the player has hand h, the dealer has hand d, and the current player is c
        return self.variables[self.variables_idx[self.get_dict_key(h, d, c)]]

    def optimize(self, display=True):
        # optimize the policy of the player
        def objective(x):
            self.variables = x
            return 1 - self.evaluate()
        
        x0 = self.variables_init
        res = optimize.minimize(objective, 
                                x0, 
                                bounds=optimize.Bounds(self.variables_lb, self.variables_ub),
                                options={'disp': display})
        self.variables_opt = res
        return res.x


class BlackjackGEKKO(Blackjack):
    # this class is a simple implementation of the game of blackjack
    # the game is played between a dealer and a player
    # the player can draw cards until they decide to stop or they bust
    # the dealer can draw cards until their score is greater than the player's score or they bust
    # the player wins if they have a higher score than the dealer or the dealer busts
    # the player loses if they bust or the dealer has a higher score
    # -----------------------------------------
    # b: the maximum score that can be achieved
    # t: the total number of cards in the deck
    # scores: the scores of the cards
    # a: the alternative score of the ace
    # q_init: the initial policy of the player
    # -----------------------------------------
    def __init__(self, b=21, t=np.array([4]*9 + [16], dtype=np.uint8), scores=np.arange(1,11), a=11, q_init=0.5, integer_q=True):
        super().__init__(b, t, scores, a, q_init)
        self.integer_q = integer_q
        self.N_variables = len(self.variables)
        self.N_states = len(self.state_dict)

        self.state_values = {}

        self.init_gekko()

    def init_states_idx(self):
        # initialize the indices of the states
        self.states_idx = [key for i, key in enumerate(self.state_dict.keys())]
        self.states_idx_dict = {key: i for i, key in enumerate(self.state_dict.keys())}

    def init_gekko_arrays(self):
        self.Arrays = []

    def init_value_cards_gekko(self, dict_key):
        # initialize the gekko variable for the value of the state key
        state_dict = self.state_dict[dict_key]

        if state_dict['num_residual_cards'] == 0:
            self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.Const(state_dict['win']))
            self.state_dict[dict_key]['Arrays_idx'] = None
            return

        if state_dict['num_cards_d'] < 1:
            self.Arrays.append(self.m.Array(self.m.SV, len(state_dict['d_plus'])))
            for i, dp in enumerate(state_dict['d_plus']):
                self.m.Equation(self.Arrays[-1][i] == self.value_cards_gekko[self.states_idx_dict[self.get_dict_key(state_dict['h'], dp, state_dict['c'])]] * state_dict['p'][i])
            self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.sum(self.Arrays[-1]))
            self.state_dict[dict_key]['Arrays_idx'] = len(self.Arrays) - 1
            return

        if state_dict['num_cards_h'] < 2:
            self.Arrays.append(self.m.Array(self.m.SV, len(state_dict['h_plus'])))
            for i, hp in enumerate(state_dict['h_plus']):
                self.m.Equation(self.Arrays[-1][i] == self.value_cards_gekko[self.states_idx_dict[self.get_dict_key(hp, state_dict['d'], state_dict['c'])]] * state_dict['p'][i])
            self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.sum(self.Arrays[-1]))
            self.state_dict[dict_key]['Arrays_idx'] = len(self.Arrays) - 1
            return
            
        
        if state_dict['score_bust_h']:
            self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.Const(0))
            self.state_dict[dict_key]['Arrays_idx'] = None
            return
        if state_dict['score_bust_d']:
            self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.Const(1))
            self.state_dict[dict_key]['Arrays_idx'] = None
            return
        
        if state_dict['c'] == 0:
            if state_dict['num_residual_cards'] == 0:
                self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.Const(state_dict['win']))
                self.state_dict[dict_key]['Arrays_idx'] = None
            elif state_dict['num_residual_cards'] == 1:
                self.m.Connection(self.value_cards_gekko[self.states_idx_dict[dict_key]], self.value_cards_gekko[self.states_idx_dict[self.get_dict_key(state_dict['h'], state_dict['d'], 1)]])
                self.state_dict[dict_key]['Arrays_idx'] = None
            else:
                self.Arrays.append(self.m.Array(self.m.SV, len(state_dict['h_plus'])))
                for i, hp in enumerate(state_dict['h_plus']):
                    self.m.Equation(self.Arrays[-1][i] == self.value_cards_gekko[self.states_idx_dict[self.get_dict_key(hp, state_dict['d'], 0)]] * state_dict['p'][i])
                self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.if3(self.q[self.variables_idx[dict_key]], self.m.sum(self.Arrays[-1]), self.value_cards_gekko[self.states_idx_dict[self.get_dict_key(state_dict['h'], state_dict['d'], 1)]]))
                print(dict_key)
                self.state_dict[dict_key]['Arrays_idx'] = len(self.Arrays) - 1
            return
        else:
            state_win = self.state_dict[dict_key]['win']
            if state_dict['num_residual_cards'] == 0:
                self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.Const(state_win))
            elif state_win == 0:
                self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.Const(state_win))
            else:
                self.m.Equation(self.value_cards_gekko[self.states_idx_dict[dict_key]] == self.m.Const(np.sum([self.value_cards(state_dict['h'], dp, state_dict['c'], lambda h,d,c: 1) * p_i for dp, p_i in zip(state_dict['d_plus'], state_dict['p'])])))
            self.state_dict[dict_key]['Arrays_idx'] = None
            return 
        
    
    def init_gekko(self):
        # initialize the gekko model
        self.m = GEKKO(remote=False)

        self.value_cards_gekko = self.m.Array(self.m.SV, self.N_states)

        #self.q = self.m.Array(self.m.Var, self.N_variables)
        #for i, q_i in enumerate(self.variables):
        #    self.q[i] = self.m.Var(value=q_i, lb=0, ub=1, integer=self.integer_q)

        self.q = self.m.Array(self.m.Var, self.N_variables, value=self.q_init, lb=0, ub=1, integer=self.integer_q)

        self.init_states_idx()
        self.init_gekko_arrays()

        for dict_key in self.states_idx:
            self.init_value_cards_gekko(dict_key)

    def optimize_gekko(self, display=True, solver=1):
        # return the value of the game when the player follows the policy q
        # solver = 1: APOPT, solver = 2: IPOPT, solver = 3: BPOPT
        self.m.Minimize(1-self.value_cards_gekko[self.states_idx_dict[self.get_dict_key(np.zeros_like(self.t), np.zeros_like(self.t), 0)]])
        self.m.options.SOLVER = solver
        self.m.solve(disp=display)
        return self.m.options.OBJFCNVAL



