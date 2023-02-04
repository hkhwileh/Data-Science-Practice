import random
import re


def no_match_intent():
    responses = ('Please tell me more.', 'Tell me more!', 'Why do you say that?', 'I see. Can you elaborate?',
                 'Interesting, Can you tell me more?', 'I see. How do you think?', 'Why?',
                 'How do you think I feel when you say that?')
    return random.choice(responses)


def cubed_intent(number):
    number = int(number)
    cubed_number = number ** 3
    return f'The cube of {number} is {cubed_number}. Isn\'t that cool?'

    # Define .no_match_intent():


def describe_planet_intent():
    responses = ('My planet is a utopia of diverse organisms and species',
                 'I am from Opidipus, the capital of Wayward Galaxies.')
    return random.choice(responses)

    # Define .answer_why_intent():


def answer_why_intent():
    responses = ('I come in peace', 'I am here to collect data on your planet and its inhabitants.',
                 'I heard the coffee is good.')
    return random.choice(responses)

    # Define .cubed_intent():


class RuleBot:
    ### Potentional Negative Response
    negative_responses = ('no', 'nope', 'nah', 'naw', 'not a chance', 'sorry')
    ### Exit conversation keyword
    exit_commands = ('quit', 'pause', 'exit', 'goodbye', 'bye', 'later')
    ### Random Starter Questions
    random_questions = (
        'Why you are here ? ',
        'Are There many human like you ?',
        'What do you consume for sustenance ?',
        'Is there Intelligent life on this planet ? ',
        'Dose Earth have a leader ?',
        'What plant you have visit ?',
        'What Technology do you have on this planet'
    )

    def __init__(self):
        self.alienbabble = {'describe_planet_intent': r'.*\s*your planet.*',
                            'answer_why_intent': r'why\sare.*',
                            'cubed_intent': r'.*cube.*(\d+)'
                            }

    # Define .greet() below:
    def greet(self):
        name = input('What is your name?')
        will_help = input(
            'Hi {}, I\'m Rashed. I\'m not from this planet. Will you help me learn about your planet?'.format(name))
        if will_help in self.negative_responses:
            print('Ok, have a nice Earth day!')
            return
        self.chat()

    # Define .make_exit() here:
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print('Ok, have a nice Earth day!')
                return True

    # define .chat next:
    def chat(self):
        reply = input(random.choice(self.random_questions)).lower()
        while not self.make_exit(reply):
            reply = input(self.match_reply(reply))

    # Define .match_reply() below:
    def match_reply(self, reply):
        for key, value in self.alienbabble.items():
            intent = key
            regex_pattern = value
            found_match = re.match(regex_pattern, reply)
            if found_match and intent == 'describe_planet_intent':
                return describe_planet_intent()
            elif found_match and intent == 'answer_why_intent':
                return answer_why_intent()
            elif found_match and intent == 'cubed_intent':
                return cubed_intent(found_match.groups()[0])
        else:
            return no_match_intent()

        # Define .describe_planet_intent():

    # Create an instance of AlienBot below:


rulechat = RuleBot()
rulechat.greet()
