import random
import torch
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from commentary2ratings.models import SeqC2R

from commentary2ratings.lm_embeddings.xlnet import XLNetHelper
commentary = []
risks = 0
choices = {0: 0, 1: 0}


def process_for_game(commentary):
    #print(commentary)
    model = XLNetHelper()
    n_samples = len(commentary)
    data = {
        'padded_commentary_embedding': [],
        'commentary_len': torch.zeros(1, dtype=torch.float32, requires_grad=False)
    }

    #for idx,row in enumerate(commentary):
    #    print(row)
    #    if idx>=n_samples:
    #        break

    data['commentary_len'][0] = n_samples
    data['padded_commentary_embedding'].append(torch.tensor(model.embed_commentaries(commentary), requires_grad=False))
    data['padded_commentary_embedding'][0] = data['padded_commentary_embedding'][0].reshape((1,data['padded_commentary_embedding'][0].shape[0], data['padded_commentary_embedding'][0].shape[1]))
    #data['padded_commentary_embedding'][0] = data['padded_commentary_embedding'][0]
    #print(data['padded_commentary_embedding'][0].shape)
    #print(len(data["padded_commentary_embedding"]),len(data["padded_commentary_embedding"][0]))
    max_len = torch.max(data['commentary_len'])
    #data['padded_commentary_embedding'] = torch.stack([F.pad(comments, (0, 0, 0, int(max_len-n_comments))) 
    #                                                for comments, n_comments in zip(data['padded_commentary_embedding'], data['commentary_len'])])

    #print(data['padded_commentary_embedding'].shape)
    #_, indices = torch.sort(data['commentary_len'], descending=True)
    
    #data = {k : data[k][0].detach().cpu() for k in data}
    data["padded_commentary_embedding"] = data["padded_commentary_embedding"][0].detach().cpu()
    data["commentary_len"] = data["commentary_len"].detach().cpu()
    #print(data["padded_commentary_embedding"])
    #print(data["padded_commentary_embedding"].shape)

    return data
            

def calculate():
    global commentary
    global risks
    data = process_for_game(commentary)
    model = SeqC2R()
    model.load_weights(15, 'experiments/rating_predictor/SeqC2R/def_norm_nll_51_nostats/weights')
    with torch.no_grad():
        model.eval()
        output = model(data).squeeze().detach().cpu().numpy()
    
    commentary = []
    print("Your Rating is: " + str(min(round(output[0] + (risks * 0.1),2),10)))
    risks = 0
    print()
    print('--------------------------------------------------')
    print()

    #print(data)

def Q3(player):

  global commentary
  global choices
  global risks
  options = ["1","2"]
  print("It all boils down to this. The final minute. Scores are level. Your last opportunity")
  print("No teammates around and defenders closing in. You have to take the shot and the only thing in your way is the goalkeeper.")
  print("You see two possible options:")
  print("1. The top right of the goal. The perfect goal, the perfect finish but difficult to execute")
  print("2. The left of the goalkeeper. The simpler shot but there is a chance the goalkeeper could reach it.")
  print("Do you go for the glory shot or the one that looks simpler? Make your decision")
  print()
  #print("Type '1' or '2'")
  userInput = ''
  while userInput not in options:
    print("Options: 1/2")
    
    userInput = input()
    print()
    if userInput == "1":

      choice = random.randint(0,1)
      np.random.choice(2,p=[0.6,0.4])
      if choice == 0:
        
        play = 'Attempt missed. ' + player + ' (Chelsea) right footed shot from the centre of the box misses to the right. Assisted by Eden Hazard with a cross.'
        commentary.append(play)

        print("An ambitious shot and on any other day that would probably be a picture-perfect goal")
        print("Unfortunately this time the ball does a little bit too much. The game ends in a draw.\nIt's not a bad result but you can't help but wonder what could have been")

      if choice == 1:
        play = "Goal!  Manchester United 1, Chelsea 2. "+player+"  - Chelsea -  shot with right foot from the centre of the box to the right corner. Assist -  Eden Hazard with a through ball following a fast break."
        commentary.append(play)
        risks += 1

        print(player.split(' ')[0].upper() +"!! YOU'VE DONE IT!! AGAINST ALL ODDS YOU SCORED THE WINNING GOAL")
        print("Congratualtions! This is surely the first of many such performances")

    elif userInput == "2":
      choice = random.randint(0,1)
      choices[choice] += 1

      if choice == 0:
        
        play =  'Attempt blocked. ' + player+ ' (Chelsea) left footed shot from the left side of the box is blocked. Assisted by Eden Hazard with a cross.'
        commentary.append(play)

        print("You go for the more plausible option")
        print("A good shot and well placed too but unfortunately the goalkeeper stretches his arm out just enough to deviate the ball.")

      if choice == 1:
        play = 'Goal!  Manchester United 1, Chelsea 2. ' + player + '  - Chelsea -  shot with right foot from the centre of the box to the left corner. Assist -  Eden Hazard.'
        commentary.append(play)

        print(player.split(' ')[0].upper() + " YOU'VE DONE IT!! AGAINST ALL ODDS YOU SCORED THE WINNING GOAL")
        print("Congratualtions! This is surely the first of many such performances")



    else: 
      print("Please enter a valid option.")

  print()
  print('--------------------------------------------------')
  print()

  calculate()



      
def Q2(player):
  global commentary
  global choices
  options = ["1","2"]
  print("Your opposition's best striker has made his way to the edges of your box with you following close by")
  print("You can probably stop what could be a goal if you try to run and tackle him but you risk a foul.")
  print("What do you do? 1. Tackle the player, 2. Play it safe and try to kick the ball away")
  print()
  userInput = ""
  while userInput not in options:
    print("Options: 1/2")
    
    userInput = input()
    print()
    if userInput == "1":

      choice = random.randint(0,1)
      choices[choice] += 1
      if choice == 0:

        play = 'Penalty conceded by ' + player + '  - Chelsea -  after a foul in the penalty area.'
        
        #play =  player + '  - Chelsea -  receive yellow card for a foul.'
        commentary.append(play)

        print("You attempt an ambitious tackle, maybe a little too ambitious...")
        print("You concede a penalty")

      if choice == 1:
        play =  'Fouled by ' +player + ' - Chelsea'
        commentary.append(play)
        commentary.append(player + '  - Chelsea -  won a free kick in defence.')

        print("You get warned for fouling the opposition.")
        print("However, you are content with yourself knowing you probably just averted a goal")

    elif userInput == "2":
      choice = random.randint(0,1)
      choices[choice] += 1

      if choice == 0:
        
        play =  player + '  - Chelsea -  receive yellow card for a foul.'
        commentary.append(play)

        print("A good attempt at stealing the ball, unfortunately you get the opposition oplayer's foot and the refree did'nt like that")
        print("You get a yellow card for a foul just outside the box.")

      if choice == 1:
        play = 'Corner -  Manchester United. Conceded by ' + player + '.'
        commentary.append(play)

        print("The opposition striker was gearing up for what looked like an easy goal...well atleast if you weren't there")
        print("You get to the ball just in time and deflect the ball out of bounds. You concede a corner but your teammates cheer for your effort!")

    else: 
      print("Please enter a valid option.")
  print()
  print('--------------------------------------------------')
  print()
  Q3(player)

def introScene(player):
  global commentary
  global choices
  options = ["1","2"]
  print("It is the 10th Minute of the match, you see two opportunities to attack:")
  print("1. You see your teammate standing unmarked on the left wing with the chance of a scoring opportunity")
  print("2. You see an opportunity to get the glory yourself but it's not as straightforward as option 1")
  print("Do you try and score yourself and make a contribution or trust your teammate with the attack?")
  print()
  #print("Type '1' or '2'")
  userInput = ''
  while userInput not in options:
    print("Options: 1/2")
    print()
    userInput = input()
    if userInput == "1":

      choice = random.randint(0,1)
      choices[choice] += 1
      if choice == 0:
        
        play = "New attacking attempt. Ãlvaro Morata  - Chelsea -  shot with right foot from outside the box is saved by goalkeeper in the centre of the goal. Assist - " + player + "."
        commentary.append(play)

        print("A good pass to your teammate allowed him to charge and shoot.\nUnfortunately it was saved by the goalkeeper")

      if choice == 1:
        play = "Goal!  Manchester United 0, Chelsea 1. Eden Hazard  - Chelsea -  shot with right foot from the centre of the box to the right corner. Assist -  " + player +  " with a through ball following a fast break."
        commentary.append(play)

        print("GOAL!! An impeccable through ball allowed your teammate to break away and a powerful right foot shot seals the deal.")

    elif userInput == "2":
      choice = random.randint(0,1)
      choices[choice] += 1

      if choice == 0:
        
        play = 'New attacking attempt. ' + player + '  - Chelsea -  shot with right foot from long distance on the left is saved in the left corner.'
        commentary.append(play)

        print("You decide to attempt to score yourself. You cut through the defenders and shoot when you see an opportunity")
        print("The goalkeeper saves the shot with a stunning dive")

      if choice == 1:
        play = 'Goal!  Manchester United 0, Chelsea 1. ' + player + '  - Chelsea -  shot with right foot from the centre of the box to the left corner. Assist -  Eden Hazard.'
        commentary.append(play)

        print("GOAL!!")
        print("You decide to attempt to score yourself. You cut through the defenders and shoot when you see an opportunity. The ball does the rest")



    else: 
      print("Please enter a valid option.")
  print()
  print('--------------------------------------------------')
  print()
  Q2(player)

if __name__ == "__main__":
  #while True:
  print("Welcome to the FIFA 2022 WC!")
  print("Playing in your debut football match as a young talent, you are excited to make a mark on the football world")
  print("You will now go through a few scenarios to get your player rating for the match")
  print("Let's start with your first name: ")
  name = input()

  print()
  print("Good luck, " +name+ ".")
  print()
  print('--------------------------------------------------')
  print()
  introScene(name)