# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def main():
    t = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1,
         1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1,
         1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1,
         1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1,
         -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1,
         1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1,
         -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1,
         -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1,
         -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
         -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

    x = pd.DataFrame(
        [[0.4016265998526575, 0.0], [0.08976093957873707, 0.0], [0.3787615964751974, 0.0], [0.39675539508732505, 0.0],
         [0.277830177207721, 0.0], [0.544543403003932, 0.01], [0.7375310516722369, 0.01], [0.5598023839425169, 0.01],
         [0.7367533827821583, 0.01], [0.7069572400164718, 0.01], [0.6351702248310199, 0.02],
         [0.20617442275367923, 0.02],
         [0.4989731564215497, 0.02], [0.3838929778431893, 0.02], [0.6231127126487606, 0.02],
         [0.40819174016520987, 0.02],
         [0.38893015905095024, 0.02], [0.6111157897939612, 0.02], [0.8505772743861436, 0.02],
         [0.8535673399567048, 0.03],
         [0.15308542030865166, 0.03], [0.7892491889142431, 0.03], [0.5701259980153122, 0.04],
         [0.6025316835444061, 0.04],
         [0.7838445728634407, 0.04], [0.37625476529480584, 0.04], [0.7083176217729235, 0.04], [0.535777478115527, 0.04],
         [0.3686783063675115, 0.04], [0.8349433137860782, 0.04], [0.3156164763275697, 0.04], [0.7506624906303683, 0.05],
         [0.46206368425412864, 0.06], [0.8754598871944277, 0.06], [0.7107502583139014, 0.06],
         [0.6469325205065573, 0.06],
         [0.7325838816293209, 0.06], [0.2828818819012416, 0.06], [0.9834187786941732, 0.06], [0.5652179502581532, 0.06],
         [0.5651609482891211, 0.07], [0.7017984308574388, 0.07], [0.21292967710443603, 0.07],
         [0.7641679416376852, 0.08],
         [0.6683492627613721, 0.08], [0.4499283513161704, 0.08], [0.24905768226977665, 0.08],
         [0.6037881587808346, 0.08],
         [0.7561007096540131, 0.08], [0.20276556503757395, 0.09], [0.2767901587846227, 0.1], [0.5752228399950394, 0.1],
         [0.445605923909008, 0.1], [0.2547662007007722, 0.1], [0.6451108310742186, 0.1], [0.6636506124316643, 0.1],
         [0.5685273317917265, 0.11], [0.67391947038051, 0.11], [0.19504030869572112, 0.11], [0.7383479548647327, 0.12],
         [0.4200853620573281, 0.12], [0.29075546022153864, 0.12], [0.1538428030774914, 0.12],
         [0.24344799567438644, 0.12],
         [0.6260749721383106, 0.12], [0.38109187818096885, 0.12], [0.5475455569979428, 0.12],
         [0.42316647150593234, 0.13],
         [0.564878005793818, 0.14], [0.9142937182458757, 0.15], [0.3697087121044591, 0.15], [0.8237868000091857, 0.15],
         [0.21157979799572954, 0.15], [0.6169111421861091, 0.15], [0.6787626611254476, 0.15],
         [0.6246361968822858, 0.15],
         [0.8428354234636216, 0.15], [0.5681829160110459, 0.15], [0.361035068031584, 0.16], [0.6862282093732724, 0.16],
         [0.5976406829318905, 0.16], [0.5880976548352763, 0.16], [0.06415850017408356, 0.16],
         [0.4884868595625844, 0.16],
         [0.44642559351661065, 0.17], [0.4956904252654863, 0.17], [0.7925988227052143, 0.17],
         [0.5891768284807015, 0.17],
         [0.37118020564567333, 0.18], [0.22898699523607297, 0.18], [0.2711112799343862, 0.18],
         [0.7521966683717558, 0.18],
         [0.09647618714657855, 0.18], [0.31502023182921507, 0.18], [0.47668431527133087, 0.19],
         [0.08016752270904978, 0.19],
         [0.8726118460419542, 0.19], [0.727607378306407, 0.19], [0.4493882371344382, 0.19], [0.6753515783630153, 0.19],
         [0.20331464895587356, 0.2], [0.6367617509138961, 0.2], [0.3803381861685349, 0.2], [0.2957652759942995, 0.2],
         [0.7522445494631051, 0.2], [0.46298497230755686, 0.2], [0.5771393651216866, 0.2], [0.34859171122918725, 0.2],
         [0.8527722485500021, 0.21], [0.702886716404516, 0.21], [0.4858177436871883, 0.21], [0.39865531893305906, 0.21],
         [0.4444553929936661, 0.21], [0.36707403612927797, 0.21], [0.4429232416188097, 0.21],
         [0.7353342384151311, 0.21],
         [0.5087034299968419, 0.21], [0.41362679058743035, 0.22], [0.6971542229289776, 0.22],
         [0.6243536197178776, 0.22],
         [0.2743824013383828, 0.22], [0.335432819704177, 0.22], [0.7483722724534199, 0.23], [0.31376176685310264, 0.23],
         [0.6535799695774651, 0.23], [0.8292872316382458, 0.24], [0.38490584944224415, 0.24],
         [0.17440881001234165, 0.24],
         [0.5997063298396508, 0.24], [0.5849945478344989, 0.25], [0.6192238731091684, 0.26], [0.6901492721429817, 0.26],
         [0.3595763940085594, 0.26], [0.7726918202519947, 0.26], [0.29018501827758475, 0.26],
         [0.5178789784700543, 0.26],
         [0.4205863469136082, 0.26], [0.1846743292995043, 0.27], [0.5876555177959288, 0.27], [0.6764824019934403, 0.27],
         [0.3411014789085116, 0.27], [0.35315182947851914, 0.27], [0.3931985539238748, 0.27],
         [0.28192538172693193, 0.27],
         [0.5678023563380692, 0.27], [0.7908503959162969, 0.28], [0.405616821394288, 0.28], [0.46294275081341446, 0.28],
         [0.4073058046822207, 0.28], [0.49552482375183776, 0.28], [0.10720029060970519, 0.28],
         [0.34462445092241933, 0.28],
         [0.4373221385026835, 0.28], [0.3893702406752945, 0.28], [0.6767564798058697, 0.29], [0.676134754506575, 0.29],
         [0.18442245400917287, 0.29], [0.5267195225163297, 0.29], [0.4908144010275756, 0.29],
         [0.6124210042949505, 0.29],
         [0.70741320007462, 0.3], [0.4809971940587671, 0.3], [0.24956216352145572, 0.31], [0.30791334209488813, 0.31],
         [0.8121631224073614, 0.31], [0.07720876159499289, 0.31], [0.45182661615230263, 0.31],
         [0.3693566041186352, 0.31],
         [0.4699266194742111, 0.31], [0.24252230456133383, 0.32], [0.5533962081939172, 0.32],
         [0.28095279442714166, 0.32],
         [0.23321530240533747, 0.32], [0.45072391577700316, 0.32], [0.4374773775717512, 0.33],
         [0.8877075289509939, 0.33],
         [0.6764938432042454, 0.33], [0.12772577326865814, 0.33], [0.6867682872969585, 0.33],
         [0.5200060762809062, 0.33],
         [0.49018354263599373, 0.34], [0.35893342122450433, 0.34], [0.2882274350700477, 0.34],
         [0.8839163584641414, 0.34],
         [0.77743958208892, 0.34], [0.9519244440195404, 0.34], [0.5830374541660774, 0.35], [0.30914361438517135, 0.35],
         [0.6509047092722482, 0.35], [0.7859573137662429, 0.35], [0.690047773680702, 0.35], [0.559578130258453, 0.35],
         [0.6831260338704924, 0.35], [0.7244688979878104, 0.36], [0.6440331411344321, 0.36],
         [0.05274806405656484, 0.36],
         [0.6082462978390231, 0.36], [0.657993583446016, 0.36], [0.3206730471804945, 0.36], [0.21748589807047514, 0.37],
         [0.536910013575228, 0.37], [0.7466925655848924, 0.38], [0.13829709386142966, 0.38], [0.6290685961012152, 0.38],
         [0.7580044581899157, 0.38], [0.31482763817701986, 0.39], [0.6845034924998591, 0.39],
         [0.33594260608926263, 0.39],
         [0.25147592190850254, 0.4], [0.6666835737993744, 0.4], [0.48208393435253255, 0.4], [0.5746788215991452, 0.4],
         [0.5624660075108279, 0.4], [0.8082936743621255, 0.4], [0.6250996092845138, 0.4], [0.21166715013051107, 0.41],
         [0.9148134652155688, 0.41], [0.4015578237523528, 0.41], [0.46631337529701533, 0.41],
         [0.5758922336609106, 0.41],
         [0.7527255882222929, 0.42], [0.3371999993211288, 0.42], [0.37194743598598545, 0.42],
         [0.3998609099969562, 0.42],
         [0.418179962960062, 0.43], [0.9584341413985813, 0.43], [0.5384141665507454, 0.43], [0.5595543597612685, 0.44],
         [0.7741294614499915, 0.44], [0.46552657208466797, 0.44], [0.30594664744024935, 0.45],
         [0.3653018540683009, 0.45],
         [0.34576189605478536, 0.45], [0.3907220382028905, 0.45], [0.2918487955120093, 0.45],
         [0.14948146118692393, 0.45],
         [0.8722092293090085, 0.45], [0.8659120385359577, 0.45], [0.42926011136087205, 0.46],
         [0.5986433513485936, 0.46],
         [0.5879392244521395, 0.46], [0.6254727123221208, 0.46], [0.689763500622278, 0.46], [0.13867759047829292, 0.46],
         [0.6873214744517182, 0.47], [0.3164616014902095, 0.48], [0.36794058079978703, 0.48],
         [0.3739093358214681, 0.48],
         [0.47694446859660355, 0.48], [0.5223532164136133, 0.48], [0.9159027029122241, 0.48],
         [0.7305551859251611, 0.48],
         [0.16055744749203138, 0.48], [0.3446914909776438, 0.49], [0.12191580957660515, 0.49],
         [0.4035578524018182, 0.49],
         [0.5251503251948173, 0.49], [0.31253510485181524, 0.5], [0.7557827738280958, 0.5], [0.5062838076499214, 0.5],
         [0.03902377259092951, 0.5], [0.6291708503250302, 0.5], [0.8092667700152653, 0.5], [0.4221872486616172, 0.5],
         [0.5518755243209977, 0.51], [0.8232855779890198, 0.51], [0.6625869665870994, 0.51], [0.5126314391194522, 0.51],
         [0.9050873927552013, 0.51], [0.15240041156084125, 0.51], [0.4637459686311767, 0.51],
         [0.10442147300396541, 0.51],
         [0.51931211411511, 0.52], [0.35359187143294313, 0.52], [0.25098749179066376, 0.52], [0.3288339736321039, 0.53],
         [0.6814048114023192, 0.54], [0.3628847797672053, 0.54], [0.5451763743262428, 0.54], [0.2590566578930637, 0.54],
         [0.7090248544163827, 0.54], [0.45781412242638336, 0.54], [0.8578701455684046, 0.55],
         [0.6988622484932335, 0.55],
         [0.21263843779251976, 0.55], [0.6489894200257244, 0.55], [0.5723673898164482, 0.56],
         [0.15114914174899619, 0.56],
         [0.687285680610243, 0.56], [0.8063128610154694, 0.56], [0.5446968185776724, 0.56], [0.6327524145059746, 0.56],
         [0.4972155140689641, 0.56], [0.7149045765798873, 0.56], [0.40645516375842994, 0.56],
         [0.43022465862795434, 0.57],
         [0.4154657713229938, 0.57], [0.48563160663804716, 0.57], [0.4498410393319968, 0.57],
         [0.07178451955396314, 0.58],
         [0.720924766693193, 0.58], [0.4176603015857099, 0.59], [0.8979330934731861, 0.6], [0.4693801450516458, 0.6],
         [0.2921698514503178, 0.61], [0.8111579490900811, 0.61], [0.44473905814114395, 0.61],
         [0.19704724437814736, 0.61],
         [0.3908714599305927, 0.62], [0.4283158861616084, 0.62], [0.5175611177268578, 0.62], [0.6681077321677007, 0.62],
         [0.5840860269448267, 0.63], [0.4325317377688, 0.63], [0.29652736272450403, 0.63], [0.7711425727210071, 0.63],
         [0.1646799629058365, 0.63], [0.4582456836614907, 0.63], [0.36198788281272215, 0.64],
         [0.9752302585171897, 0.64],
         [0.4935200300051664, 0.64], [0.6830346497195462, 0.64], [0.5018911259533252, 0.65],
         [0.31822268578046764, 0.65],
         [0.09329428428255389, 0.65], [0.6666339845569527, 0.65], [0.5061723279413584, 0.65],
         [0.6240328072645426, 0.65],
         [0.40965701782077646, 0.65], [0.5983722510926639, 0.65], [0.9340759288031006, 0.66],
         [0.3868015564344795, 0.66],
         [0.6550344503832947, 0.66], [0.29900403777193707, 0.67], [0.4326578031520352, 0.67],
         [0.7565474165807863, 0.67],
         [0.6453585400996804, 0.68], [0.561986925485203, 0.69], [0.46665159415569524, 0.7], [0.6021605329339057, 0.7],
         [0.4732271230328498, 0.7], [0.5776862137054486, 0.71], [0.5763141811199117, 0.71], [0.18811978587290928, 0.71],
         [0.8438803109484488, 0.71], [0.4720008761567981, 0.71], [0.4243072909772645, 0.72], [0.6780093350838861, 0.72],
         [0.6389028819701059, 0.72], [0.46232993012974927, 0.72], [0.43573785679135885, 0.73],
         [0.04284952064061896, 0.73],
         [0.8476965173017345, 0.73], [0.47202573973965084, 0.73], [0.03757768005290599, 0.73],
         [0.7016283958666147, 0.73],
         [0.32350419179090295, 0.73], [0.5682407971494237, 0.73], [0.577363363306987, 0.73], [0.7473635324325578, 0.74],
         [0.7261025462217732, 0.74], [0.6551845532009123, 0.74], [0.29157364446737516, 0.74],
         [0.49665864628780015, 0.74],
         [0.34393072950761205, 0.74], [0.3022849238616449, 0.74], [0.18904553325840517, 0.75],
         [0.7250989681123179, 0.75],
         [0.9193400693663425, 0.75], [0.5998990891211389, 0.75], [0.1589123897684065, 0.75], [0.3605282553360045, 0.75],
         [0.15394832925579716, 0.75], [0.21552491825499295, 0.75], [0.7109209838378698, 0.76],
         [0.44707138702784216, 0.76],
         [0.439153750238975, 0.76], [0.49803582986087647, 0.77], [0.5855280587053193, 0.77],
         [0.48008646606264954, 0.77],
         [0.5119839558779391, 0.77], [0.6446496623250454, 0.77], [0.3423452482731689, 0.77], [0.5184414650064146, 0.77],
         [0.8071262409831651, 0.77], [0.14536916238726477, 0.78], [0.707499641408291, 0.78], [0.5366073439835419, 0.78],
         [0.5256753810807246, 0.78], [0.446726439305412, 0.78], [0.1295106087601432, 0.79], [0.2724712734264805, 0.79],
         [0.8457308300861378, 0.79], [0.5209780735917655, 0.79], [0.5426162974193625, 0.8], [0.5140399070258648, 0.8],
         [0.7186734183423144, 0.8], [0.27410406775657575, 0.8], [0.4578726918721119, 0.8], [0.3299474277523871, 0.8],
         [0.6110791777960136, 0.81], [0.7606765115429113, 0.81], [0.7191772013244511, 0.81],
         [0.24113826902386815, 0.81],
         [0.5647276644156408, 0.81], [0.5658378859048487, 0.81], [0.6148573917145737, 0.82], [0.6856891697316678, 0.82],
         [0.11855985442926958, 0.83], [0.29573228122647593, 0.83], [0.21183511804753452, 0.83],
         [0.3055839956802341, 0.83],
         [0.8825458103485998, 0.83], [0.5387714824212761, 0.83], [0.3946914945125458, 0.84], [0.3039066979849908, 0.84],
         [0.7762186294799708, 0.84], [0.6070365213339483, 0.84], [0.9120364296910444, 0.84], [0.6312254304409697, 0.85],
         [0.4513103999280518, 0.86], [0.866209838687191, 0.86], [0.1459841045851169, 0.86], [0.3548046825022322, 0.87],
         [0.27782068387349934, 0.87], [0.4985948477958673, 0.88], [0.7499559571362844, 0.88],
         [0.5544309899505286, 0.88],
         [0.5782872868557781, 0.89], [0.6331708457582477, 0.89], [0.9031958059326254, 0.89], [0.6437649509766373, 0.9],
         [0.3989116966545191, 0.9], [0.368666343152542, 0.9], [0.2967983521039811, 0.9], [0.5022480066621907, 0.9],
         [0.7083845818407222, 0.9], [0.6377938188231502, 0.91], [0.393156893954662, 0.91], [0.5770938366402103, 0.91],
         [0.31572542034114154, 0.91], [0.2480480958235495, 0.91], [0.30166923028720916, 0.92],
         [0.5428754771491575, 0.92],
         [0.5722009483779962, 0.93], [0.527504459668444, 0.93], [0.5586679811211115, 0.94], [0.8461292826003396, 0.94],
         [0.951922288868373, 0.94], [0.15565574709731667, 0.94], [0.29102228884771764, 0.94],
         [0.7174851716068088, 0.94],
         [0.439669403444407, 0.94], [0.6181670932193537, 0.95], [0.14906183535032097, 0.95], [0.2711841075020769, 0.95],
         [0.3467386406122351, 0.95], [0.5095784328937327, 0.95], [0.8543215358881662, 0.96], [0.2893184862502555, 0.96],
         [0.16278471402169978, 0.97], [0.6404521944869057, 0.97], [0.21100667282763025, 0.97],
         [0.4501111469638733, 0.98],
         [0.33566440950532594, 0.98], [0.6942167306612723, 0.98], [0.68446951251531, 0.98], [0.3739761290184217, 0.98],
         [0.39607553102283266, 0.98], [0.30332093982257263, 0.98], [0.882281803288737, 0.98],
         [0.6257608794822735, 0.99],
         [0.324351769331815, 0.99]])

    b = pd.DataFrame(np.ones(len(x)))
    x = pd.concat([b, x], axis=1)

    x.columns = ["b", "x1", "x2"]

    w = pd.DataFrame([1.0, 2.0, 3.0])

    p = 0.5

    for j in range(100):
        for i in range(len(x)):
            print np.dot(x.iloc[i], w) * t[i]

            # if np.dot(x,w) < 0:
            #   w = w + p*np.dot(x,w)*x
