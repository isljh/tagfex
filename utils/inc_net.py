import copy
from copy import deepcopy
import logging
import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear, TagFex_SimpleLinear
try:
    from torchvision.ops import MLP
except ImportError:
    MLP = None


def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained,args=args)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained,args=args)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained,args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        self.convnet.load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )

class IL2ANet(IncrementalNet):

    def update_fc(self, num_old, num_total, num_aux):
        fc = self.generate_fc(self.feature_dim, num_total+num_aux)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
        del self.fc
        self.fc = fc

class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer_BIC(nn.Module):
    def __init__(self):
        super(BiasLayer_BIC, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer_BIC())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data

            # --- 核心修复区域：数据不在一个设备 ---
            # 1. 获取模型权重所在的设备
            target_device = weight.device

            if nextperiod_initialization is not None:
                # 确保初始化张量也在目标设备
                weight = torch.cat([weight, nextperiod_initialization.to(target_device)])
            else:
                # 2. 将新的零向量创建在目标设备上
                new_zeros = torch.zeros(nb_classes - nb_output, self.feature_dim).to(target_device)
                weight = torch.cat([weight, new_zeros])
            # --- 修复结束 ---
            '''
            if nextperiod_initialization is not None:

                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            '''
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def regenerate_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        del self.fc
        self.fc = fc
        return fc

class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        # no need the convnet.

        print(
            'Clear the convnet in MultiBranchCosineIncrementalNet, since we are using self.convnets with dual branches')
        self.convnet = torch.nn.Identity()
        for param in self.convnet.parameters():
            param.requires_grad = False

        self.convnets = nn.ModuleList()
        self.args = args

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        # import pdb; pdb.set_trace()
        out = self.fc(features)
        out.update({"features": features})
        return out

    def construct_dual_branch_network(self, trained_model, tuned_model, cls_num):
        self.convnets.append(trained_model.convnet)
        self.convnets.append(tuned_model.convnet)

        self._feature_dim = self.convnets[0].out_dim * len(self.convnets)
        self.fc = self.generate_fc(self._feature_dim, cls_num)


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc
    

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x , bias=True):
        ret_x = x.clone()
        ret_x = (self.alpha+1) * x # + self.beta
        if bias:
            ret_x = ret_x + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())



class TagFexNet(nn.Module):
    def __init__(self, args, pretrained):
        super(TagFexNet, self).__init__()
        self.convnet_type = args["convnet_type"] #模型的backbone:resnet18
        self.convnets = nn.ModuleList()  # Task-specific Models (fts) 的集合
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

        self._device = args["device"][0]
        self.ta_net = get_convnet(args).to(self._device)  # Task-agnostic Model (fta)
        #hasattr判断“一个对象有没有某个属性”
        if hasattr(self.ta_net, 'fc'):
            self.ta_net.fc = None

        self.fusion_type = self.args.get("fusion_type", "ts_attention")
        self.replace_ta_with_task0_ts = self.args.get("replace_ta_with_task0_ts", False)
        self.task0_ta_substitute = None
        self.ts_attn = None  # Merge Attention 模块
        self.fusion_linear = None
        self.trans_classifier = None  # 融合特征分类器，用于计算图中的 L_mcls
        #将 TA 特征映射到高维空间，用于计算 L_ta
        self.projector = nn.Sequential(
            TagFex_SimpleLinear(self.ta_feature_dim, self.args["proj_hidden_dim"]),
            nn.BatchNorm1d(self.args["proj_hidden_dim"]),
            nn.ReLU(True),
            TagFex_SimpleLinear(self.args["proj_hidden_dim"], self.args["proj_hidden_dim"]),
            nn.BatchNorm1d(self.args["proj_hidden_dim"]),
            nn.ReLU(True),
            TagFex_SimpleLinear(self.args["proj_hidden_dim"], self.args["proj_output_dim"]),
            nn.BatchNorm1d(self.args["proj_output_dim"]),
        ).to(self._device)
        #-----------------------------------------------------------------------
        #self.proj = MLP(512, [2048, 2048, 16], norm_layer=nn.BatchNorm1d)
        #---------------------------------------------------------------------------
        self.predictor = None

    @property
    def ta_feature_dim(self):
        return self._get_active_ta_model().out_dim #512

    #计算fs最后拼接完的特征长度
    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    #拼接特征
    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        x.to(self._device)
        ts_outs = [convnet(x) for convnet in self.convnets]
        features = [ts_out["features"] for ts_out in ts_outs]
        features = torch.cat(features, 1)

        #out = self.fc(features)  # {logics: self.fc(features)}
        out = {}
        out["logits"] = self.fc(features)
        aux_logits = self.aux_fc(features[:, -self.out_dim :])

        out.update({"aux_logits": aux_logits, "features": features})
        #取出最后一层的特征图
        ta_model = self._get_active_ta_model()
        ta_fmap = ta_model(x)['fmaps'][-1] # (bs, C, H, W)  [bs, 512, 4, 4]
        ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1) # (bs, H*W, C) -mean-> (bs, C)  (bs, 512)
        #print(ta_feature)
        #assert 0
        embedding = self.projector(ta_feature)

        out.update({
                'ta_feature': ta_feature,
                'embedding': embedding,
        })

        if self.trans_classifier is not None:
            ts_feature = ts_outs[-1]["fmaps"][-1].flatten(2).permute(0, 2, 1)
            ta_features = ta_fmap.flatten(2).permute(0, 2, 1)# (bs, H*W, C) 
            merged_feature = self._merge_ta_ts_features(ta_features.detach(), ts_feature)
            trans_logits = self.trans_classifier(merged_feature)
            out.update(trans_logits=trans_logits)

        if self.predictor is not None:
            predicted_feature = self.predictor(ta_feature)
            out.update(predicted_feature=predicted_feature)            
        
        return out
        """
        {
            'ta_feature': ta_feature,                      ta分支得到的特征
            'embedding': embedding,                        自监督嵌入,ta分支得到的特征经过projector得到的嵌入
            'trans_logits': trans_logits,                  融合后的特征进行分类
            'predicted_feature': predicted_feature,        服务于 知识蒸馏,根据当前的 ta_feature 去“预测”旧模型提取出来的特征
            'features': features                           所有任务特定专家提取特征的总和
            'logits': logits                               合并特征分类
            'aux_logits':aux_logits                        辅助分支分类（新旧类别）
        }
        """
     
    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args).to(self._device))
        else:
            self.convnets.append(get_convnet(self.args).to(self._device))
            #新 task 的 TS 模型不是随机初始化，而是：旧 TS 模型 + TA 模型 的加权融合
            #init_interpolation_factor:初始插值系数
            init_interpolation_factor = self.args["init_interpolation_factor"]
            ta_model = self._get_active_ta_model()
            for ts_old_parameter, ts_new_parameter, ta_prarameter in zip(self.convnets[-2].parameters(), self.convnets[-1].parameters(), ta_model.parameters()):
                ts_new_parameter.data = init_interpolation_factor * ts_old_parameter.data + (1 - init_interpolation_factor) * ta_prarameter.data

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1).to(self._device)

        """new added"""
        if self.predictor is None:
            self.predictor = self.generate_fc(self.ta_feature_dim, self.ta_feature_dim).to(self._device)
        if self.fusion_type == "ts_attention":
            if self.ts_attn is None:
                self.ts_attn = TSAttention(self.out_dim, self.args['attn_num_heads'], device=self._device)
            else:
                self.ts_attn._reset_parameters()
        elif self.fusion_type == "concat_linear":
            if self.fusion_linear is None:
                self.fusion_linear = TagFex_SimpleLinear(self.out_dim * 2, self.out_dim).to(self._device)
            else:
                self.fusion_linear.reset_parameters()
        elif self.fusion_type == "mean":
            pass
        else:
            raise ValueError("Unknown fusion_type: {}".format(self.fusion_type))
        self.trans_classifier = self.generate_fc(self.ta_feature_dim, new_task_size).to(self._device)

        if self.replace_ta_with_task0_ts and self.task0_ta_substitute is None and len(self.convnets) > 0:
            self.task0_ta_substitute = deepcopy(self.convnets[0]).to(self._device)
            for p in self.task0_ta_substitute.parameters():
                p.requires_grad_(False)
            self.task0_ta_substitute.eval()

    def generate_fc(self, in_dim, out_dim):
        fc = TagFex_SimpleLinear(in_dim, out_dim).to(self._device)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    def get_freezed_copy_ta(self):
        from copy import deepcopy
        ta_net_copy = deepcopy(self._get_active_ta_model()) 
        for p in ta_net_copy.parameters():
            p.requires_grad_(False)
        return ta_net_copy.eval()

    def get_freezed_copy_projector(self):
        from copy import deepcopy
        projector_copy = deepcopy(self.projector)
        for p in projector_copy.parameters():
            p.requires_grad_(False)
        return projector_copy.eval()
    
    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def _merge_ta_ts_features(self, ta_features, ts_feature):
        if self.fusion_type == "ts_attention":
            return self.ts_attn(ta_features, ts_feature).mean(1)
        if self.fusion_type == "concat_linear":
            merged = torch.cat([ta_features, ts_feature], dim=-1)
            return self.fusion_linear(merged).mean(1)
        if self.fusion_type == "mean":
            return (ta_features + ts_feature).mean(1)
        raise ValueError("Unknown fusion_type: {}".format(self.fusion_type))

    def _get_active_ta_model(self):
        if self.replace_ta_with_task0_ts and self.task0_ta_substitute is not None:
            return self.task0_ta_substitute
        return self.ta_net

    #权重对齐
    def weight_align(self, increment):
        if increment <= 0:
            logging.info("Skip weight alignment because increment is {}.".format(increment))
            return
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class TagFexSimplifiedNet(nn.Module):
    def __init__(self, args, pretrained):
        super(TagFexSimplifiedNet, self).__init__()
        self.args = args
        self.pretrained = pretrained
        self._device = args["device"][0]

        self.convnets = nn.ModuleList()
        self.ta_net = get_convnet(args).to(self._device)
        if hasattr(self.ta_net, "fc"):
            self.ta_net.fc = None

        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.projector = nn.Sequential(
            TagFex_SimpleLinear(self.ta_feature_dim, self.args["proj_hidden_dim"]),
            nn.BatchNorm1d(self.args["proj_hidden_dim"]),
            nn.ReLU(True),
            TagFex_SimpleLinear(self.args["proj_hidden_dim"], self.args["proj_hidden_dim"]),
            nn.BatchNorm1d(self.args["proj_hidden_dim"]),
            nn.ReLU(True),
            TagFex_SimpleLinear(self.args["proj_hidden_dim"], self.args["proj_output_dim"]),
            nn.BatchNorm1d(self.args["proj_output_dim"]),
        ).to(self._device)
        self.predictor = None

    @property
    def ta_feature_dim(self):
        return self.ta_net.out_dim

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        return torch.cat(features, 1)

    def forward(self, x):
        ts_outs = [convnet(x) for convnet in self.convnets]
        features = [ts_out["features"] for ts_out in ts_outs]
        features = torch.cat(features, 1)

        out = {
            "logits": self.fc(features),
            "features": features,
        }
        out["aux_logits"] = self.aux_fc(features[:, -self.out_dim :])

        ta_fmap = self.ta_net(x)["fmaps"][-1]
        ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)
        out["ta_feature"] = ta_feature
        out["embedding"] = self.projector(ta_feature)

        if self.predictor is not None:
            out["predicted_feature"] = self.predictor(ta_feature)

        return out

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args).to(self._device))
        else:
            self.convnets.append(get_convnet(self.args).to(self._device))
            init_interpolation_factor = self.args["init_interpolation_factor"]
            for ts_old_parameter, ts_new_parameter, ta_parameter in zip(
                self.convnets[-2].parameters(),
                self.convnets[-1].parameters(),
                self.ta_net.parameters(),
            ):
                ts_new_parameter.data = (
                    init_interpolation_factor * ts_old_parameter.data
                    + (1 - init_interpolation_factor) * ta_parameter.data
                )

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1).to(self._device)

        if self.predictor is None:
            self.predictor = self.generate_fc(self.ta_feature_dim, self.ta_feature_dim).to(self._device)

    def generate_fc(self, in_dim, out_dim):
        return TagFex_SimpleLinear(in_dim, out_dim).to(self._device)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def get_freezed_copy_ta(self):
        ta_net_copy = deepcopy(self.ta_net)
        for p in ta_net_copy.parameters():
            p.requires_grad_(False)
        return ta_net_copy.eval()

    def get_freezed_copy_projector(self):
        projector_copy = deepcopy(self.projector)
        for p in projector_copy.parameters():
            p.requires_grad_(False)
        return projector_copy.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma


class TagFexTask0TSReplaceNet(nn.Module):
    def __init__(self, args, pretrained, use_transfer_branch=True):
        super(TagFexTask0TSReplaceNet, self).__init__()
        self.args = args
        self.pretrained = pretrained
        self._device = args["device"][0]
        self.use_transfer_branch = use_transfer_branch

        self.convnets = nn.ModuleList()
        self.task0_ta_substitute = None
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.ts_attn = None
        self.trans_classifier = None

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    @property
    def ta_feature_dim(self):
        if self.out_dim is not None:
            return self.out_dim
        if self.task0_ta_substitute is not None:
            return self.task0_ta_substitute.out_dim
        return 0

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        return torch.cat(features, 1)

    def _build_task0_ta_substitute(self):
        if self.task0_ta_substitute is None and len(self.convnets) > 0:
            self.task0_ta_substitute = deepcopy(self.convnets[0]).to(self._device)
            for p in self.task0_ta_substitute.parameters():
                p.requires_grad_(False)
            self.task0_ta_substitute.eval()

    def _get_ta_substitute(self):
        if self.task0_ta_substitute is not None:
            return self.task0_ta_substitute
        return None

    def forward(self, x):
        ts_outs = [convnet(x) for convnet in self.convnets]
        features = [ts_out["features"] for ts_out in ts_outs]
        features = torch.cat(features, 1)

        out = {
            "logits": self.fc(features),
            "features": features,
        }
        out["aux_logits"] = self.aux_fc(features[:, -self.out_dim :])

        if self.use_transfer_branch and self.trans_classifier is not None and self.task0_ta_substitute is not None:
            with torch.no_grad():
                ta_fmap = self.task0_ta_substitute(x)["fmaps"][-1]
            ts_feature = ts_outs[-1]["fmaps"][-1].flatten(2).permute(0, 2, 1)
            ta_features = ta_fmap.flatten(2).permute(0, 2, 1)
            merged_feature = self.ts_attn(ta_features, ts_feature).mean(1)
            out["trans_logits"] = self.trans_classifier(merged_feature)

        return out

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args).to(self._device))
        else:
            self._build_task0_ta_substitute()
            self.convnets.append(get_convnet(self.args).to(self._device))
            init_interpolation_factor = self.args["init_interpolation_factor"]
            ta_model = self.task0_ta_substitute
            for ts_old_parameter, ts_new_parameter, ta_parameter in zip(
                self.convnets[-2].parameters(),
                self.convnets[-1].parameters(),
                ta_model.parameters(),
            ):
                ts_new_parameter.data = (
                    init_interpolation_factor * ts_old_parameter.data
                    + (1 - init_interpolation_factor) * ta_parameter.data
                )

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1).to(self._device)

        if self.use_transfer_branch:
            if self.ts_attn is None:
                self.ts_attn = TSAttention(self.out_dim, self.args["attn_num_heads"], device=self._device)
            else:
                self.ts_attn._reset_parameters()
            self.trans_classifier = self.generate_fc(self.out_dim, new_task_size).to(self._device)

    def generate_fc(self, in_dim, out_dim):
        return TagFex_SimpleLinear(in_dim, out_dim).to(self._device)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma


class TACLSNet(nn.Module):
    def __init__(self, args, pretrained=False):
        super(TACLSNet, self).__init__()
        self.args = args
        self.pretrained = pretrained
        self._device = args["device"][0]

        self.ta_net = get_convnet(args).to(self._device)
        if hasattr(self.ta_net, "fc"):
            self.ta_net.fc = None

        self.fc = None
        self.projector = nn.Sequential(
            TagFex_SimpleLinear(self.ta_feature_dim, self.args["proj_hidden_dim"]),
            nn.BatchNorm1d(self.args["proj_hidden_dim"]),
            nn.ReLU(True),
            TagFex_SimpleLinear(self.args["proj_hidden_dim"], self.args["proj_hidden_dim"]),
            nn.BatchNorm1d(self.args["proj_hidden_dim"]),
            nn.ReLU(True),
            TagFex_SimpleLinear(self.args["proj_hidden_dim"], self.args["proj_output_dim"]),
            nn.BatchNorm1d(self.args["proj_output_dim"]),
        ).to(self._device)
        self.predictor = None

    @property
    def ta_feature_dim(self):
        return self.ta_net.out_dim

    @property
    def feature_dim(self):
        return self.ta_feature_dim

    def extract_vector(self, x):
        ta_fmap = self.ta_net(x)["fmaps"][-1]
        return ta_fmap.flatten(2).permute(0, 2, 1).mean(1)

    def forward(self, x):
        x.to(self._device)
        ta_out = self.ta_net(x)
        ta_fmap = ta_out["fmaps"][-1]
        ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)
        embedding = self.projector(ta_feature)

        out = {
            "ta_feature": ta_feature,
            "embedding": embedding,
            "logits": self.fc(ta_feature),
        }

        if self.predictor is not None:
            out["predicted_feature"] = self.predictor(ta_feature)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        if self.predictor is None:
            self.predictor = self.generate_fc(self.ta_feature_dim, self.ta_feature_dim).to(self._device)

    def generate_fc(self, in_dim, out_dim):
        return TagFex_SimpleLinear(in_dim, out_dim).to(self._device)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def get_freezed_copy_ta(self):
        from copy import deepcopy
        ta_net_copy = deepcopy(self.ta_net)
        for p in ta_net_copy.parameters():
            p.requires_grad_(False)
        return ta_net_copy.eval()

    def get_freezed_copy_projector(self):
        from copy import deepcopy
        projector_copy = deepcopy(self.projector)
        for p in projector_copy.parameters():
            p.requires_grad_(False)
        return projector_copy.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma


class TACLSDetachNet(TACLSNet):
    def __init__(self, args, pretrained=False):
        super(TACLSDetachNet, self).__init__(args, pretrained)

    def forward(self, x):
        x.to(self._device)
        ta_out = self.ta_net(x)
        ta_fmap = ta_out["fmaps"][-1]
        ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)
        embedding = self.projector(ta_feature)

        out = {
            "ta_feature": ta_feature,
            "embedding": embedding,
            "logits": self.fc(ta_feature.detach()),
        }

        if self.predictor is not None:
            out["predicted_feature"] = self.predictor(ta_feature)

        return out

class TSAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        #创建两个 LayerNorm 模块
        self.norm_ts = nn.LayerNorm(embed_dim, device=device)
        self.norm_ta = nn.LayerNorm(embed_dim, device=device)

        self.weight_q = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_k_ts = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_k_ta = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_v_ts = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_v_ta = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
    
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.weight_q)
        nn.init.xavier_normal_(self.weight_k_ts)
        nn.init.xavier_normal_(self.weight_k_ta)
        nn.init.xavier_normal_(self.weight_v_ts)
        nn.init.xavier_normal_(self.weight_v_ta)

        self.norm_ta.reset_parameters()
        self.norm_ts.reset_parameters()
    
    def forward(self, ta_feats, ts_feats):
        bs, N, C = ta_feats.shape
        # feats: (bs, N, C)
        ta_feats = self.norm_ta(ta_feats)
        ts_feats = self.norm_ts(ts_feats)

        q = (ts_feats @ self.weight_q).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2) # (bs, H, N, Ch)
        k_ts = (ts_feats @ self.weight_k_ts).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k_ta = (ta_feats @ self.weight_k_ta).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_ts = (ts_feats @ self.weight_v_ts).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_ta = (ta_feats @ self.weight_v_ta).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        feat = nn.functional.scaled_dot_product_attention(q, torch.cat((k_ta, k_ts), dim=2), torch.cat((v_ta, v_ts), dim=2)) # (bs, H, N, Ch) # use default scale

        feat = feat.transpose(1, 2).flatten(2)

        return feat


class DERNet_TagFex(nn.Module):
    """
    消融实验用的纯 DER 网络，去掉了 TagFex 中的：
    - ta_net（任务无关网络）
    - projector（自监督投影头）
    - TSAttention（融合注意力）
    - trans_classifier（迁移分类器）
    - predictor（预测头）

    只保留：
    - convnets（任务特定网络列表）
    - fc（主分类器）
    - aux_fc（辅助分类器）
    """

    def __init__(self, args, pretrained=False):
        super(DERNet_TagFex, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        ts_outs = [convnet(x) for convnet in self.convnets]
        features = [ts_out["features"] for ts_out in ts_outs]
        features = torch.cat(features, 1)

        out = {}
        out["logits"] = self.fc(features)
        aux_logits = self.aux_fc(features[:, -self.out_dim:])
        out.update({"aux_logits": aux_logits, "features": features})

        return out
        """
        {
            'logits': logits,
            'aux_logits': aux_logits,
            'features': features,
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            # Task 0：直接随机初始化
            self.convnets.append(get_convnet(self.args).to(self._device))
        else:
            # Task 1+：从旧 convnet 复制初始化（纯 DER 风格，不依赖 ta_net）
            self.convnets.append(get_convnet(self.args).to(self._device))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim

        # 更新主分类器，保留旧权重
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

        # 更新辅助分类器
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1).to(self._device)

    def generate_fc(self, in_dim, out_dim):
        return TagFex_SimpleLinear(in_dim, out_dim).to(self._device)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights, gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma
