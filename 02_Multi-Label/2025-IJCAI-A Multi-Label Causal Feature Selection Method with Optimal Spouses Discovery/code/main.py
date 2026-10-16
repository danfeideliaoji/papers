from  MCF_Sp import MCF_Sp
import matlab.engine
import matlab
import scipy.io as sio
eng = matlab.engine.start_matlab()

if __name__ == "__main__":
    """An example of MCF_Spouse"""

    dataset = sio.loadmat("data\HumanGO-train.mat")
    data = dataset["train"]
    alpha = 0.05 # Significance level
    L = 14 # number of labels
    k1 = 0.1
    selfea = MCF_Sp(data,alpha,L,k1)
    print(selfea)
