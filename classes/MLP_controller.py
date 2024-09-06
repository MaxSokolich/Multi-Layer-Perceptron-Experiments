import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io
import random
import torch
import numpy as np
from torch import nn
import copy
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation


class MLP(nn.Module):

    def __init__(self, nInput, nOutput):
        super(MLP, self).__init__()

        # two hidden layers
        self.layers = nn.Sequential(
            nn.Linear(nInput, 40),
            nn.ReLU(),
            nn.Linear(40, 60),
            nn.ReLU(),
            nn.Linear(60, nOutput)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Interp1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes
        torch.searchsorted(v['x'].contiguous().squeeze(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)



class MLP_controller(object):

    def __init__(self,mlp_model_a = None,mlp_model_b = None,a_means=None,b_means = None,dt = 1) -> None:

        self.N = 1  #descritization
        self.interp1d = Interp1d.apply


        if mlp_model_a is None or mlp_model_b is None:
            nInput = 2
            nOutput = 2+2+2
            model_a = MLP(nInput, nOutput)
            model_b = MLP(nInput, nOutput)
            model_a.load_state_dict(torch.load('models/model_a_510.pt'))
            model_b.load_state_dict(torch.load('models/model_b_510.pt'))
            self.model_a = model_a
            self.model_b = model_b
        if a_means is None or b_means is None:
            train_data_a = scipy.io.loadmat('models/data_a_2_510.mat')
            train_data_b = scipy.io.loadmat('models/data_b_2_510.mat')
            self.a_means = torch.tensor(train_data_a['means'])
            self.b_means = torch.tensor(train_data_b['means'])
            self.FREQS = torch.tensor(train_data_b['freqs'])
            self.MEAN_SPEED = torch.tensor(train_data_b['speeds'])
        pass
        self.dt = dt
        self.shuffle_flag = False


    def sim(self, robot1_start, robot1_end, robot2_start, robot2_end, ctrl):

      
        #test with a random da and db

        #limits for robot position
        XMIN = 10
        XMAX = 60
        YMIN = 20
        YMAX = 70


        a0 = torch.tensor(robot1_start)
        a1 = torch.tensor(robot1_end)
        b0 = torch.tensor(robot2_start)
        b1 = torch.tensor(robot2_end)
        
        
        # a0 = torch.tensor([(XMAX-XMIN)*np.random.rand() + XMIN, (YMAX-YMIN)*np.random.rand() + YMIN])
        # a1 = torch.tensor([(XMAX-XMIN)*np.random.rand() + XMIN, (YMAX-YMIN)*np.random.rand() + YMIN])
        # b0 = torch.tensor([(XMAX-XMIN)*np.random.rand() + XMIN, (YMAX-YMIN)*np.random.rand() + YMIN])
        # b1 = torch.tensor([(XMAX-XMIN)*np.random.rand() + XMIN, (YMAX-YMIN)*np.random.rand() + YMIN])

        start_a, = plt.plot(a0[0].detach().numpy(), a0[1].detach().numpy(), color = 'r', marker = 's', label = 'ubot A start')
        goal_a, = plt.plot(a1[0].detach().numpy(), a1[1].detach().numpy(), color = 'r', marker = 'x', label = 'ubot A goal')
        start_b, = plt.plot(b0[0].detach().numpy(), b0[1].detach().numpy(), color = 'b', marker = 's', label = 'ubot B start')
        goal_b, = plt.plot(b1[0].detach().numpy(), b1[1].detach().numpy(), color = 'b', marker = 'x', label = 'ubot B goal')

        da = a1 - a0
        db = b1 - b0
        print(da)
        print(db)

        print("ctrl = ",ctrl)
        #ctrl = torch.tensor(ctrl).unsqueeze(-1)

        speeds_a = self.interp1d(self.FREQS, self.MEAN_SPEED[0,:], torch.round(ctrl[:,0:2])).squeeze()
        speeds_b = self.interp1d(self.FREQS, self.MEAN_SPEED[1,:], torch.round(ctrl[:,0:2])).squeeze()
        angles = ctrl[:,2:4]
        deltaTimes = ctrl[:,4:6]

        delta_a = torch.zeros(2)
        delta_b = torch.zeros(2)

        delta_a[0] = torch.sum(speeds_a * torch.cos(angles) * deltaTimes) #x
        delta_a[1] = torch.sum(speeds_a * torch.sin(angles) * deltaTimes) #y
        delta_b[0] = torch.sum(speeds_b * torch.cos(angles) * deltaTimes) #x
        delta_b[1] = torch.sum(speeds_b * torch.sin(angles) * deltaTimes) #y

        numSteps = 1
        deltaTimes = deltaTimes / numSteps
        pos_a = np.zeros((2,2*numSteps+1))
        pos_b = np.zeros((2,2*numSteps+1))
        pos_a[:,0] = a0
        pos_b[:,0] = b0
        for i in range(1,numSteps+1):
            pos_a[0,2*i-1] = pos_a[0,2*i-2] + speeds_a[0] * torch.cos(angles[0,0]) * deltaTimes[0,0] #x
            pos_a[1,2*i-1] = pos_a[1,2*i-2] + speeds_a[0] * torch.sin(angles[0,0]) * deltaTimes[0,0] #y
            pos_b[0,2*i-1] = pos_b[0,2*i-2] + speeds_b[0] * torch.cos(angles[0,0]) * deltaTimes[0,0] #x
            pos_b[1,2*i-1] = pos_b[1,2*i-2] + speeds_b[0] * torch.sin(angles[0,0]) * deltaTimes[0,0] #y

            pos_a[0,2*i] = pos_a[0,2*i-1] + speeds_a[1] * torch.cos(angles[0,1]) * deltaTimes[0,1] #x
            pos_a[1,2*i] = pos_a[1,2*i-1] + speeds_a[1] * torch.sin(angles[0,1]) * deltaTimes[0,1] #y
            pos_b[0,2*i] = pos_b[0,2*i-1] + speeds_b[1] * torch.cos(angles[0,1]) * deltaTimes[0,1] #x
            pos_b[1,2*i] = pos_b[1,2*i-1] + speeds_b[1] * torch.sin(angles[0,1]) * deltaTimes[0,1] #y

        plt.scatter(pos_a[0,:], pos_a[1,:], color = 'r', marker = '.')
        plt.scatter(pos_b[0,:], pos_b[1,:], color = 'b', marker = '.')
        traj1, = plt.plot(pos_a[0,:], pos_a[1,:], color = 'r', marker = '.', label = 'ubot A')
        traj2, = plt.plot(pos_b[0,:], pos_b[1,:], color = 'b', marker = '.', label = 'ubot B')

        plt.legend(handles=[start_a, goal_a, start_b, goal_b])
        
        plt.ylim([0, 2048])
        plt.xlim([0, 2448])
        plt.show()
        print(delta_a)
        print(delta_b)


    def getControl(self,Dx):#da, db):

        da = Dx[0,:]
        db = Dx[1,:]
        mag_a = torch.sqrt(torch.dot(da,da))
        mag_b = torch.sqrt(torch.dot(db,db))
        if mag_a > mag_b:
            angle = torch.arctan2(da[1], da[0]) - torch.pi/2;
            #print(angle)
            A = torch.tensor([[torch.cos(-angle), -torch.sin(-angle)], [torch.sin(-angle), torch.cos(-angle)]]) / mag_a
            db = A @ db
            #print(db)

            reflect = False
            if db[0] < 0:
                db[0] = -db[0]
                reflect = True

            #print(db)
            ctrl = self.model_a(db.float()).double() + self.a_means
            #print(ctrl)
            if reflect:
                #print(ctrl[:,2:4])
                #print(torch.sin(ctrl[:,2:4]))
                #print(torch.cos(ctrl[:,2:4]))
                ctrl[:,2:4] = torch.arctan2(torch.sin(ctrl[:,2:4]), -torch.cos(ctrl[:,2:4]))
                #print(ctrl)

            ctrl[:,2:4] += angle
            ctrl[:,4:6] *= mag_a

            freqs, alphas = self.discretization(ctrl) 

        else:
            angle = torch.arctan2(db[1], db[0]) - torch.pi/2;
            A = torch.tensor([[torch.cos(-angle), -torch.sin(-angle)], [torch.sin(-angle), torch.cos(-angle)]]) / mag_b
            da = A @ da

            reflect = False
            if da[0] < 0:
                da[0] = -da[0]
                reflect = True


            ctrl = self.model_b(da.float()).double() + self.b_means

            if reflect:
                ctrl[:,2:4] = torch.arctan2(torch.sin(ctrl[:,2:4]), -torch.cos(ctrl[:,2:4]))

            ctrl[:,2:4] += angle
            ctrl[:,4:6] *= mag_b

            freqs, alphas = self.discretization(ctrl) 
            
        if self.shuffle_flag:
            # Create an array of indices
            indices = np.arange(len(freqs))

            # Shuffle the indices
            np.random.shuffle(indices)

            # Use the shuffled indices to shuffle the array
            freqs = freqs[indices]
            alphas = alphas[indices] 

        return freqs, alphas, ctrl

    def discretization(self,u,Nc = 10):
        
        
        print("u = ", u)
        u = u[0]


        DTs_time1 = u[-2].item()/self.dt
        DTs_time2 = u[-1].item()/self.dt
        


        freqs1 = np.repeat(u[0].item(), DTs_time1)
        print("\n\nfreqs1 = ", freqs1)
        freqs2 = np.repeat(u[1].item(), DTs_time2)
        alphas1 = np.repeat(u[2].item(), DTs_time1)
        alphas2 = np.repeat(u[3].item(), DTs_time2)

        freqs_array = np.array(list(freqs1) + list(freqs2))

        alphas_array = np.array(list(alphas1) + list(alphas2))

        print("freqs arr = ", freqs_array)
        print("DTs = ", DTs_time1,DTs_time2)
        
        # if not self.shuffle_flag:
        #     freqs = self.merge_elments(elm1 = u[0].item(), elm2 = u[1].item(), T1 = DTs_time1 , T2 = DTs_time2, N = 5)
        #     alphas =  self.merge_elments(elm1 = u[2].item(), elm2 = u[3].item(), T1 = DTs_time1 , T2 = DTs_time2, N = 5)
        # else: 
        #     freqs = np.zeros(int(DTs_time1+DTs_time2))
        #     alphas = np.zeros(int(DTs_time1+DTs_time2))
        #     # frequincy sequence:
        #     freqs[0:int(DTs_time1)] = u[0].item()
        #     freqs[int(DTs_time1+1):int(DTs_time1+DTs_time2)] = u[1].item()
        #     # heading sequence:  
        #     alphas[0:int(DTs_time1)] = u[2].item()
        #     alphas[int(DTs_time1+1):int(DTs_time1+DTs_time2)] = u[3].item()

        return freqs_array, alphas_array 



    def merge_elments(self,elm1, elm2, T1, T2, N = 5):
   
        if T1>T2: 
            x2 = T1/T2
            e1 = True
            assert T2 > N, 'The length of the 2nd control input seqeunce is smaller than the requisted N_rep'
            n1 = int(x2 * N) 
            n2 = N
            rep = int(T2 / N)
        else: 
            x2 = T2/T1
            e1 = False
            assert T1 > N, 'The length of the 1st control input seqeunce is smaller than the requisted N_rep'
            n1 = N 
            n2 = int(x2 * N)
            rep = int(T1 / N) 
        resd1 = T1 - n1 * rep
        resd2 = T2 - n2 *rep  
        u1_resd = list(np.repeat(elm1,resd1))
        u2_resd = list(np.repeat(elm1,resd2))
        u1 = list(np.repeat(elm1,n1))
        u2 = list(np.repeat(elm2,n2))
        [u1.append(elem) for elem in u2]
        # Now do the concatnation: 
        u = list(np.tile(u1,rep))
        [u.append(elem) for elem in u1_resd]
        [u.append(elem) for elem in u2_resd]
        return np.array(u) 
# ---------------------------





def main(): 

    mlpC = MLP_controller()
    mlpC.discretization(u  = np.array([1,2,.1,.2,5,6]))  


if __name__ == '__main__':
    main()