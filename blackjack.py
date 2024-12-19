import numpy as np
from scipy import optimize
import copy

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
        score1 = self.Adiff * min([np.floor((self.b-score0)/self.Adiff), u[0]]) # the score of the ace
        return score0 + score1
    
    def score_bust(self, score_u):
        # return whether the score is greater than the maximum score
        return score_u > self.b
    
    def prob_draw(self, h, d):
        # return the probability of drawing a card
        r = self.t - h - d
        r_total = np.sum(r)
        return r/r_total
    
    def plus_one(self, u, i):
        # add one card i to the hand
        u[i] += 1
        return u

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

        if state_dict['num_cards_h'] < 2:
            self.state_values[dict_key] = np.sum([self.value_cards(hp, d, c, q) * p_i for hp, p_i in zip(state_dict['h_plus'], state_dict['p'])])
            return self.state_values[dict_key]
        
        if state_dict['num_cards_d'] < 1:
            self.state_values[dict_key] = np.sum([self.value_cards(h, dp, c, q) * p_i for dp, p_i in zip(state_dict['d_plus'], state_dict['p'])])
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
            state_value_1 = np.sum([self.value_cards(hp, d, 0, q) * p_i for hp, p_i in zip(state_dict['h_plus'], state_dict['p'])])
            self.state_values[dict_key] = rate * state_value_0 + (1-rate) * state_value_1
        else:
            rate = q(h,d,c)
            state_value_0 = self.def_win(h, d)
            state_value_1 = np.sum([self.value_cards(h, dp, 1, q) * p_i for dp, p_i in zip(state_dict['d_plus'], state_dict['p'])])
            self.state_values[dict_key] = rate * state_value_0 + (1-rate) * state_value_1
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
        state_dict['h_plus'] = self.draw_results(h, state_dict['p'])
        state_dict['d_plus'] = self.draw_results(d, state_dict['p'])
        state_dict['score_h'] = self.score_cards(h)
        state_dict['score_d'] = self.score_cards(d)
        state_dict['score_bust_h'] = self.score_bust(state_dict['score_h'])
        state_dict['score_bust_d'] = self.score_bust(state_dict['score_d'])
        state_dict['win'] = self.def_win(h, d)
        state_dict['num_cards_h'] = self.num_cards(h)
        state_dict['num_cards_d'] = self.num_cards(d)
        
        self.state_dict[dict_key] = state_dict

        if state_dict['num_cards_h'] < 2:
            for hp in state_dict['h_plus']:
                self.init_hdc(hp, d, c)
            return 
        
        if state_dict['num_cards_d'] < 1:
            for dp in state_dict['d_plus']:
                self.init_hdc(h, dp, c)
            return 

        if state_dict['score_bust_h']:
            return 
        if state_dict['score_bust_d']:
            return 
        
        self.variables_idx[dict_key] = len(self.variables)
        self.variables.append(self.q_init)
        
        if c == 0:
            for hp in state_dict['h_plus']:
                self.init_hdc(hp, d, 0)
            for dp in state_dict['d_plus']:
                self.init_hdc(h, dp, 1)
            return
        else:
            for dp in state_dict['d_plus']:
                self.init_hdc(h, dp, 1)
            return
        
    def get_variable(self, h, d, c):
        # return the policy of the player when the player has hand h, the dealer has hand d, and the current player is c
        return self.variables[self.variables_idx[self.get_dict_key(h, d, c)]]

    def optimize(self, display=True):
        # optimize the policy of the player
        def objective(x):
            for i, v in enumerate(x):
                self.variables[i] = v
            return 1 - self.evaluate()
        
        x0 = self.variables_init
        res = optimize.minimize(objective, 
                                x0, 
                                bounds=optimize.Bounds(self.variables_lb, self.variables_ub),
                                options={'disp': display})
        self.variables_opt = res.x
        return res.x
