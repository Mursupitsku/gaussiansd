import math
import copy
from scipy.stats import norm

__all__ = [
    # Gaussian score difference objects
    'GaussianSD', 'Rating',
    # functions for the global environment
    'rate'
    # default values
    'MU', 'SIGMA', 'BETA', 'TAU', 'SD_VARIANCE',
]


#: Default mean
MU = 25.
#: Default standard definition
SIGMA = MU / 3
#: Default distance that guarantees about 76% chance of winning.
BETA = SIGMA / 2
#: Default dynamic factor.
TAU = SIGMA / 100
#: Default score difference variance
SD_VARIANCE = 10

def to_pi(sigma):
    """
    Converts the standard deviation, sigma, of gaussian distribution to precision, pi
    """
    return 1/sigma**2

def to_tau(pi, mu):
    return pi*mu

def to_sigma(pi):
    return math.sqrt(1/pi)

def to_mu(tau, pi):
    return tau/pi

def divide_gaussian(mu1, sigma1, mu2, sigma2):
    pi1 = to_pi(sigma1)
    tau1 = to_tau(pi1, mu1)
    
    pi2 = to_pi(sigma2)
    tau2 = to_tau(pi2, mu2)
    
    new_pi = pi1 - pi2
    new_tau = tau1 - tau2
    
    new_sigma = math.sqrt(1/new_pi)
    new_mu = new_tau / new_pi
    
    return new_mu, new_sigma

def sum_gaussians(gaussians, coeffs = None):
    
    if coeffs is None:
        coeffs = []
        for _ in len(gaussians):
            coeffs.append(1)
    
    new_mu = 0
    variance = 0
    
    for gaussian, coeff in zip(gaussians, coeffs):
        new_mu += coeff*gaussian['mu']
        variance += coeff**2*gaussian['sigma']**2
    
    new_sigma = math.sqrt(variance)
    return new_mu, new_sigma
    
def multiply_gaussian(mu1, sigma1, mu2, sigma2):
    pi1 = 1/sigma1**2
    tau1 = pi1*mu1
    
    pi2 = 1/sigma2**2
    tau2 = pi2*mu2
    
    new_pi = pi1 + pi2
    new_tau = tau1 + tau2
    
    new_sigma = math.sqrt(1/new_pi)
    new_mu = new_tau / new_pi
    
    return new_mu, new_sigma

class Rating():
    """
    
    """
    
    def __init__(self, mu=None, sigma=None):
        if isinstance(mu, tuple):
            mu, sigma = mu
        if mu is None:
            mu = global_env().mu
        if sigma is None:
            sigma = global_env().sigma
        self.mu = mu
        self.sigma = sigma
        
    def __repr__(self):
        c = type(self)
        #print(c)
        args = ('.'.join([c.__module__, c.__name__]), self.mu, self.sigma)
        #print(args)
        return '%s(mu=%.3f, sigma=%.3f)' % args

    
class GaussianSD():
    """
    
    """
    
    def __init__(self, mu=MU, sigma=SIGMA, beta=BETA, tau=TAU,
                 score_difference_variance=SD_VARIANCE):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.tau = tau
        self.score_difference_variance = score_difference_variance

    def rate(self, team1, team2, team1_score, team2_score):
        """
        
        """
        
        team1_priors = []
        team2_priors = []
        
        for team, team_priors in zip([team1, team2], [team1_priors, team2_priors]):
        
            for player in team:
                
                team_priors.append({'mu': player.mu, 'sigma': player.sigma})
        
        #Prior factor
        for team in [team1_priors, team2_priors]:
            for player in team:
                player['sigma'] = math.sqrt(self.tau**2 + player['sigma']**2)
        #print(team1_priors, team2_priors)
        
        #Likelihood factor down
        team1_down_msg = []
        team2_down_msg = []
    
        for team, msg in zip([team1_priors, team2_priors], [team1_down_msg, team2_down_msg]):
        
            for player in team:
            
                pi = to_pi(player['sigma'])
                tau = to_tau(pi, player['mu'])
                a = 1 / (1 + self.beta**2 * pi)
                sigma = to_sigma(a*pi)
                mu = to_mu(a*tau, a*pi)
            
                msg.append({'mu': mu, 'sigma': sigma})
        #print(team1_down_msg, team2_down_msg)
        
        #Sum factor down
        team1_sum = {'mu': 0, 'sigma': 0}
        team2_sum = {'mu': 0, 'sigma': 0}
    
        for msg, sum in zip([team1_down_msg, team2_down_msg], [team1_sum, team2_sum]):
            variance = 0
            for player in msg:
                sum['mu'] += player['mu']
                variance += player['sigma']**2
            sum['sigma'] = math.sqrt(variance)
    
        #print(team1_sum, team2_sum)
        
        #Inference
        team1_updated = {'mu': 0, 'sigma': 0}
        team2_updated = {'mu': 0, 'sigma': 0}
        
        teams = [team1_sum, team2_sum]
        opponents = [team2_sum, team1_sum]
        scores = [team1_score, team2_score]
        opponent_scores = [team2_score, team1_score]
        updated_ratings = [team1_updated, team2_updated]
        
        for team, opponent, team_score, opponent_score, updated_rating in zip(teams, opponents, scores, opponent_scores, updated_ratings):
            
            size = len(team1) + len(team2)
            updated_pi = 1/team['sigma']**2 + 1/(2*self.beta**2 + self.score_difference_variance**2 + opponent['sigma']**2)
            updated_tau = team['mu']/team['sigma']**2 + (team_score - opponent_score + opponent['mu'])/(2*self.beta**2 + self.score_difference_variance**2 + opponent['sigma']**2)
            
            
            #print(updated_pi)
            #print(updated_tau)
            
            updated_sigma = to_sigma(updated_pi)
            updated_mu = to_mu(updated_tau, updated_pi)
            
            updated_rating['mu'] = updated_mu
            updated_rating['sigma'] = updated_sigma
        
        #print(team1_updated, team2_updated)
        
        team1_up_msg = {}
        team2_up_msg = {}
        
        team1_up_msg['mu'], team1_up_msg['sigma'] = divide_gaussian(team1_updated['mu'], team1_updated['sigma'], team1_sum['mu'], team1_sum['sigma'])
        team2_up_msg['mu'], team2_up_msg['sigma'] = divide_gaussian(team2_updated['mu'], team2_updated['sigma'], team2_sum['mu'], team2_sum['sigma'])
        #print(team1_up_msg, team2_up_msg)
        
        team1_up_msgs = []
        team2_up_msgs = []
    
        for team, msg, up_msgs in zip([team1_down_msg, team2_down_msg], [team1_up_msg, team2_up_msg], [team1_up_msgs, team2_up_msgs]):
            #print(team)
            #print(msg)
            for player in team:
            
                up_player = copy.deepcopy(msg)
            
                for other_player in team:
                    if other_player is player:
                        continue
                
                    up_player['mu'], up_player['sigma'] = sum_gaussians([up_player, other_player], [1,-1])
            
                up_msgs.append(up_player)
        
        #print(team1_up_msgs, team2_up_msgs)
        
        team1_posteriors = []
        team2_posteriors = []
    
        for team, msg, posteriors in zip([team1_priors, team2_priors], [team1_up_msgs, team2_up_msgs], [team1_posteriors, team2_posteriors]):
        
            for player, msg_player in zip(team, msg):
                #print(player, msg_player)
                pi = to_pi(msg_player['sigma'])
                #print(pi)
                tau = to_tau(pi, msg_player['mu'])
                a = 1 / (1 + self.beta**2 * pi)
                #print(a)
            
                sigma = to_sigma(a*pi)
                mu = to_mu(a*tau, a*pi)
                #print(mu, sigma)
            
                player_posterior = {}
                player_posterior['mu'], player_posterior['sigma'] = multiply_gaussian(mu, sigma, player['mu'], player['sigma'])
            
                posteriors.append(player_posterior)
        
        #print(team1_posteriors, team2_posteriors)
        
        team1_posterior_ratings = []
        team2_posterior_ratings = []
        
        for posteriors, posterior_ratings in zip([team1_posteriors, team2_posteriors], [team1_posterior_ratings, team2_posterior_ratings]):
            
            for posterior in posteriors:
                
                posterior_ratings.append(Rating(mu = posterior['mu'], sigma = posterior['sigma']))
        
        #team1_posteriors = (Rating(mu=99), Rating(mu=99))
        #print(team1_posteriors)
        #team2_posteriors = (Rating(mu=99), Rating(mu=99))
        #print(team2_posteriors)
        
        return team1_posterior_ratings, team2_posterior_ratings
        
    def win_probability2(self, team1, team2):
        
        team1_ratings = []
        team2_ratings = []
        
        for team, team_ratings in zip([team1, team2], [team1_ratings, team2_ratings]):
        
            for player in team:
                
                team_ratings.append({'mu': player.mu, 'sigma': player.sigma})
        
        team1_down_msg = []
        team2_down_msg = []
    
        for team, msg in zip([team1_ratings, team2_ratings], [team1_down_msg, team2_down_msg]):
        
            for player in team:
            
                pi = to_pi(player['sigma'])
                tau = to_tau(pi, player['mu'])
                a = 1 / (1 + self.beta**2 * pi)
                sigma = to_sigma(a*pi)
                mu = to_mu(a*tau, a*pi)
            
                msg.append({'mu': mu, 'sigma': sigma})
        #print(team1_down_msg, team2_down_msg)
        
        #Sum factor down
        team1_sum = {'mu': 0, 'sigma': 0}
        team2_sum = {'mu': 0, 'sigma': 0}
    
        for msg, sum in zip([team1_ratings, team2_ratings], [team1_sum, team2_sum]):
            variance = 0
            for player in msg:
                sum['mu'] += player['mu']
                variance += player['sigma']**2
            sum['sigma'] = math.sqrt(variance)
        
        #print(team1_sum, team2_sum)
        p = 1 - norm.cdf((team1_sum['mu'] - team2_sum['mu'])/(team1_sum['sigma']**2 + team2_sum['sigma']**2 + 2*self.beta**2))
        return p
        
    def win_probability(self, team1, team2):
        delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
        #sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
        sum_sigma = sum(r.sigma**2 for r in team1) + sum(r.sigma**2 for r in team2)
        #print(sum_sigma)
        size = len(team1) + len(team2)
        denom = math.sqrt(size * (self.beta * self.beta) + sum_sigma)
        return norm.cdf(delta_mu / denom)
        

def global_env():
    """Gets the :class:`TrueSkill` object which is the global environment."""
    try:
        global_env.__gaussiansd__
    except AttributeError:
        # setup the default environment
        setup()
    return global_env.__gaussiansd__


def setup(mu=MU, sigma=SIGMA, beta=BETA, tau=TAU, score_difference_variance=SD_VARIANCE, env=None):
    """Setups the global environment.

    :param env: the specific :class:`TrueSkill` object to be the global
                environment.  It is optional.

    >>> Rating()
    trueskill.Rating(mu=25.000, sigma=8.333)
    >>> setup(mu=50)  #doctest: +ELLIPSIS
    trueskill.TrueSkill(mu=50.000, ...)
    >>> Rating()
    trueskill.Rating(mu=50.000, sigma=8.333)

    """
    if env is None:
        env = GaussianSD(mu, sigma, beta, tau, score_difference_variance)
    global_env.__gaussiansd__ = env
    return env
    
