from dependencies import *
from helperFunctions import *

class Net(nn.Module):
    """
    The class that defines the graph architecture of the DNN to be used. Definition of convolutional
    layers, fully connected layers, batch-normalization operations, etc. are created here.

    Args:
        None: Network definition within the initialization routine.

    Returns:
        Instantiates a Net object, with a member function forward_prop. The forward_prop member function
        takes in the input signal and returns the output signal at the network’s output.
    """
    def __init__(self, arch, w_init_scheme='He', bias_inits=0.0, incep_layers=-1, nEmbed=-1, nClasses=-1):
        super(Net, self).__init__()
 
        # Initialization scheme
        self.weightsInitsScheme = w_init_scheme
        self.biasesInitTo = bias_inits
        self.N_incep_layers = incep_layers
        # Save off the selected DNN architecture as a member.
        self.arch = arch    

        # Dictionary of supported DNN architectures.
        switcher = {
            'cnn_108x108': self.cnn_108x108, 
            'inceptionModuleV1_108x108': self.inceptionModuleV1_108x108,
            'inceptionModuleV1_75x45': self.inceptionModuleV1_75x45,
            'inceptionTwoModulesV1_75x45': self.inceptionTwoModulesV1_75x45,
            'inceptionTwoModulesV1_root1_75x45': self.inceptionTwoModulesV1_root1_75x45,
            'inceptionV1_modularized': self.inceptionV1_modularized,
            'inceptionV1_modularized_mnist': self.inceptionV1_modularized_mnist,
            'centerlossSimple': self.centerlossSimple
        }
    
        # Select the net definition given by arch.
        netDefinition = switcher.get(arch)
    
        # Initialize the architecture selected.
        try:
            if self.arch == 'centerlossSimple':
                assert(nEmbed > 0)
                assert(nClasses > 0)
                netDefinition(nEmbed, nClasses)
            else:
                netDefinition()
        
            print("DNN arch:", self.arch)
        except Exception as e:
            print("Specified DNN architecture not implemented. Error:", e)
            sys.exit()

        # Show percentage of each layer's parameters relative to the total.
        self.show_layer_parameter_percentages()
            
        # Initialize the DNN layers with the specified scheme.
        self.initialize_layers()

    ######################################## Member functions ########################################
    def show_layer_parameter_percentages(self):
        # Compute the total number of learnable parameters.
        paramIterator = list(self.parameters())
        self.N_dnnParameters = 0
        for pp in paramIterator:                
            if len(pp.size()) == 1:
                self.N_dnnParameters += pp.size()[0]
            else:
                self.N_dnnParameters += np.prod(pp.size())

        # Show number of learnable parameters per module.
        self.N_runningParams = 0        
        for m in self.modules():
            if isinstance(m, Net):
                continue
            elif isinstance(m, nn.Sequential):
                print("\n")
                continue        
            elif isinstance(m, nn.ModuleList):
                print("\n")
                continue                    
            else:
                params = list(m.parameters())
                N_lenParams = len(params)
                N_currentParams = 0            
                if N_lenParams > 0:
                    for pp in range(N_lenParams):
                        N_currentParams += np.prod(params[pp].size())
                else:
                    N_currentParams = 0

                self.N_runningParams += N_currentParams                
                print("Module: {} params: {:.2f}".format(m.__class__.__name__, 100.0 * N_currentParams / float(self.N_dnnParameters)))

        print("\n")
        print("Total number of trainable parameters: {:5d}\n".format(self.N_dnnParameters))
        assert(self.N_runningParams == self.N_dnnParameters)
        
    def initialize_layers(self):
        print("Initializing layers via: {}, biases to: {:.2f}".format(self.weightsInitsScheme, self.biasesInitTo))
    
        # Initialize selected layers.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Compute the fan-in.
                if isinstance(m, nn.Conv2d):
                    fanIn = m.in_channels * m.weight.size()[2] * m.weight.size()[3]
                elif isinstance(m, nn.Linear):
                    fanIn = m.in_features

                print("Module: {} FanIn: {:1d}".format(m.__class__.__name__, fanIn))

                # Initialize weights.
                if self.weightsInitsScheme == 'He':
                    m.weight.data.normal_(0, np.sqrt(2.0 / fanIn))

                # Initialize biases.
                try:
                    m.bias.data.fill_(self.biasesInitTo)
                except Exception as e:
                    print("No biases found for this layer. Error:", e)

    def forward(self, x):
        if self.arch == 'cnn_108x108':
            x = self.cnn(x)
            x = x.view(-1, self.num_flat_features(x))
            x = self.classifier(x)
            return x

        elif self.arch in ['inceptionModuleV1_108x108', 'inceptionModuleV1_75x45']:
            root = self.root(x)
            b_1x1 = self.b_1x1(root)
            b_3x3 = self.b_3x3(root)
            b_5x5 = self.b_5x5(root)
            b_pool = self.b_pool(root)
            concat = torch.cat((b_1x1, b_3x3, b_5x5, b_pool), 1)
            redux = self.redux(concat)
            redux = redux.view(-1, self.num_flat_features(redux))
            logits = self.classifier(redux)
            return logits

        elif self.arch == 'inceptionTwoModulesV1_75x45':
            root = self.root(x)
            b_1x1 = self.b_1x1(root)
            b_3x3 = self.b_3x3(root)
            b_5x5 = self.b_5x5(root)
            b_pool = self.b_pool(root)
            concat = torch.cat((b_1x1, b_3x3, b_5x5, b_pool), 1)
            b2_1x1 = self.b2_1x1(concat)
            b2_3x3 = self.b2_3x3(concat)
            b2_5x5 = self.b2_5x5(concat)
            b2_pool = self.b2_pool(concat)
            concat2 = torch.cat((b2_1x1, b2_3x3, b2_5x5, b2_pool), 1)
            redux = self.redux(concat2)
            redux = redux.view(-1, self.num_flat_features(redux))
            logits = self.classifier(redux)
            return logits

        elif self.arch == 'inceptionTwoModulesV1_root1_75x45':
            root = self.root(x)
            b_1x1 = self.b_1x1(root)
            b_3x3 = self.b_3x3(root)
            b_5x5 = self.b_5x5(root)
            b_pool = self.b_pool(root)
            concat = torch.cat((b_1x1, b_3x3, b_5x5, b_pool), 1)
            b2_1x1 = self.b2_1x1(concat)
            b2_3x3 = self.b2_3x3(concat)
            b2_5x5 = self.b2_5x5(concat)
            b2_pool = self.b2_pool(concat)
            concat2 = torch.cat((b2_1x1, b2_3x3, b2_5x5, b2_pool), 1)
            redux = self.redux(concat2)
            redux = redux.view(-1, self.num_flat_features(redux))
            logits = self.classifier(redux)
            return logits

        elif self.arch in ['inceptionV1_modularized', 'inceptionV1_modularized_mnist']:
            root = self.root(x)
            for ii in range(self.N_incep_layers):
                if ii > 0:
                    root = incepOut
                for bb in range(len(self.masterList[ii])):
                    temp = self.masterList[ii][bb](root)
                    if bb == 0:
                        incepOut = temp
                    else:
                        incepOut = torch.cat((incepOut, temp), 1)
            redux = self.redux(incepOut)
            redux = redux.view(-1, self.num_flat_features(redux))
            self.x = self.fc1(redux)
            logits = self.fc2(self.x)
            return logits

        elif self.arch == 'centerlossSimple':
            root = self.root(x)
            root = root.view(-1, self.num_flat_features(root))
            self.x = self.latent(root)
            logits = self.logits(self.x)
            return logits

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def centerlossSimple(self, nEmbed, nClasses):
        self.centroids = torch.from_numpy(0.01*np.random.randn(nEmbed, nClasses)).cuda(0)
        self.root = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=(1,1), padding=(2,2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, stride=(1,1), padding=(2,2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(32, 64, 5, stride=(1,1), padding=(2,2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=(1,1), padding=(2,2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(64, 128, 5, stride=(1,1), padding=(2,2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, stride=(1,1), padding=(2,2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.latent = nn.Linear(1152, nEmbed, bias=True)
        self.logits = nn.Linear(nEmbed, nClasses, bias=False)

    def inceptionV1_modularized_mnist(self):
        self.centroids = torch.from_numpy(np.random.randn(2, 10)).cuda(0)
        if not isinstance(self.N_incep_layers, int) or self.N_incep_layers <= 0:
            print("Selected inceptionV1_modularized, but number of inception layers wanted is less than 0.")
            sys.exit()
        self.root = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(64, 256, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.masterList = nn.ModuleList()
        for ii in range(self.N_incep_layers):
            incep = nn.ModuleList()
            incep += self.create_inception_module_v1()
            self.masterList += [incep]
        self.redux = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(32, 16, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(16, 4, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(36, 2)
        self.fc2 = nn.Linear(2, 10, bias=False)

    def inceptionV1_modularized(self):
        if not isinstance(self.N_incep_layers, int) or self.N_incep_layers <= 0:
            print("Selected inceptionV1_modularized, but number of inception layers wanted is less than 0.")
            sys.exit()
        self.root = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(64, 256, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.masterList = nn.ModuleList()
        for ii in range(self.N_incep_layers):
            incep = nn.ModuleList()
            incep += self.create_inception_module_v1()
            self.masterList += [incep]
        self.redux = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(32, 16, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(16, 4, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(180, 2)
        )

    def create_inception_module_v1(self):
        b_1x1 = nn.Sequential(
            nn.Conv2d(256, 64, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        b_3x3 = nn.Sequential(
            nn.Conv2d(256, 96, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        b_5x5 = nn.Sequential(
            nn.Conv2d(256, 16, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=(1,1), padding=(2,2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        b_pool = nn.Sequential(
            nn.MaxPool2d((3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(256, 32, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        return [b_1x1, b_3x3, b_5x5, b_pool]

    def cnn_108x108(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(8, 16, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(16, 32, 3, stride=(1,1), padding=(0,0)),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(32, 64, 3, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(64, 64, 3, stride=(3,3), padding=(0,0)),
            nn.Dropout2d(),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.Linear(16, 2)
        )

    def inceptionModuleV1_108x108(self):
        self.root = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(64, 192, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.b_1x1 = nn.Sequential(
            nn.Conv2d(192, 64, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b_3x3 = nn.Sequential(
            nn.Conv2d(192, 96, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b_5x5 = nn.Sequential(
            nn.Conv2d(192, 16, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=(1,1), padding=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.b_pool = nn.Sequential(
            nn.MaxPool2d((3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(192, 32, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.redux = nn.Sequential(
            nn.Conv2d(256, 64, 2, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(64, 32, 1, stride=(1,1), padding=(0,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(32, 16, 1, stride=(1,1), padding=(0,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)),
            nn.Conv2d(16, 4, 1, stride=(1,1), padding=(0,0)),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(36, 2)
        )

    def inceptionModuleV1_75x45(self):
        self.root = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(64, 192, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.b_1x1 = nn.Sequential(
            nn.Conv2d(192, 64, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b_3x3 = nn.Sequential(
            nn.Conv2d(192, 96, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b_5x5 = nn.Sequential(
            nn.Conv2d(192, 16, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=(1,1), padding=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.b_pool = nn.Sequential(
            nn.MaxPool2d((3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(192, 32, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.redux = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(32, 16, 3, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(16, 4, 1, stride=(1,1), padding=(0,0)),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 2)
        )

    def inceptionTwoModulesV1_75x45(self):
        self.root = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(64, 192, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.b_1x1 = nn.Sequential(
            nn.Conv2d(192, 64, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b_3x3 = nn.Sequential(
            nn.Conv2d(192, 96, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b_5x5 = nn.Sequential(
            nn.Conv2d(192, 16, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=(1,1), padding=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.b_pool = nn.Sequential(
            nn.MaxPool2d((3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(192, 32, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.b2_1x1 = nn.Sequential(
            nn.Conv2d(256, 64, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b2_3x3 = nn.Sequential(
            nn.Conv2d(256, 96, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b2_5x5 = nn.Sequential(
            nn.Conv2d(256, 16, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=(1,1), padding=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.b2_pool = nn.Sequential(
            nn.MaxPool2d((3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(256, 32, 1, stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.redux = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(32, 16, 3, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(16, 4, 1, stride=(1,1), padding=(0,0)),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 2)
        )

    def inceptionTwoModulesV1_root1_75x45(self):
        self.root = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(64, 192, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.b_1x1 = nn.Sequential(
            nn.Conv2d(192, 64, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b_3x3 = nn.Sequential(
            nn.Conv2d(192, 96, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b_5x5 = nn.Sequential(
            nn.Conv2d(192, 16, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=(1,1), padding=(2,2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.b_pool = nn.Sequential(
            nn.MaxPool2d((3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(192, 32, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.b2_1x1 = nn.Sequential(
            nn.Conv2d(256, 64, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b2_3x3 = nn.Sequential(
            nn.Conv2d(256, 96, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b2_5x5 = nn.Sequential(
            nn.Conv2d(256, 16, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=(1,1), padding=(2,2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.b2_pool = nn.Sequential(
            nn.MaxPool2d((3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(256, 32, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.redux = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(32, 16, 3, stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(16, 4, 1, stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(180, 2)
        )
