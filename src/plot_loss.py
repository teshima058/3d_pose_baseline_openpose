import matplotlib.pyplot as plt

def display_loss(log_file):
    epoch = []
    lr = []
    loss_train = []
    loss_test = []
    err = []

    with open(log_file, 'r') as log:
        for i,l in enumerate(log):
            line = l.rstrip()
            line = line.split(',')
            epoch.append(int(line[0]))
            lr.append(float(line[1]))
            loss_train.append(float(line[2]))
            loss_test.append(float(line[3]))
            err.append(float(line[4]))

    plt.plot(loss_train, color='#FF0000', label='Training Loss')
    plt.plot(loss_test, color='#004AFF', label='Test Loss')

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.show()

def main():
    log_file = './log/train.log'
    display_loss(log_file)

if __name__=='__main__':
    main() 