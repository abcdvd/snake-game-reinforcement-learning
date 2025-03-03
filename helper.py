import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, mean_loss):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(mean_loss)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(mean_loss)-1, mean_loss[-1], str(mean_loss[-1]))

    plt.show(block=False)
    plt.pause(.1)


def plot_loss(losses, mean_losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training - losses')
    plt.xlabel('Number of Games')
    plt.ylabel('Losses')
    plt.plot(losses)
    plt.plot(mean_losses)
    plt.ylim(ymin=0)
    if len(losses) > 0:
        plt.text(len(losses)-1, losses[-1], f"{losses[-1]:.4f}")
    if len(mean_losses) > 0:
        plt.text(len(mean_losses)-1, mean_losses[-1], f"{mean_losses}")
    plt.show(block=False)
    plt.pause(.1)