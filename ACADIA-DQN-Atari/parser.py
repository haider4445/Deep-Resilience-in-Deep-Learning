import argparse
def parser():

	parser = argparse.ArgumentParser(description = "Fast Undetectable Attack")
	parser.add_argument('-mp','--Path', metavar = 'path', type = str, help = 'Complete path to model')
	parser.add_argument('-e','--env', type = str, nargs = "?", default = "PongNoFrameskip-v4", help = 'Environment name like PongNoFrameskip-v4')
	parser.add_argument('-p','--perturbationType', nargs="?", default="rfgsm", type = str, help = 'Perturbation Type: fgsm, rfgsm, cw, optimal')
	parser.add_argument('-a', '--attack', nargs="?", default=1, type = int, help = 'Attack 1 or not to attack 0')
	parser.add_argument('--stepsRFGSM', nargs = "?", default = 20, type = int, help = "Number of steps of RFGSM attack")
	parser.add_argument('--alphaRFGSM', nargs = "?", default = 2/255, type = float, help = "Alpha (Step Size) of RFGSM attack")
	parser.add_argument('--epsRFGSM', nargs = "?", default = 8/255, type = float, help = "Epsilon (strength) of RFGSM attack")
	parser.add_argument('--totalgames', nargs = "?", default = 10, type = int, help = "total games/episodes")
	parser.add_argument('--strategy', nargs = "?", default = "allSteps", type = str, help = "Attack strategy: random, allSteps, leastSteps, critical")
	parser.add_argument('--targeted', nargs = "?", default = 0, type = int, help = "0 or 1")
	parser.add_argument('--defended', nargs = "?", default = 0, type = int, help = "-1 (non-defended DRL agent) or 1 (RADIAL)")	

	parser.add_argument('--stepsFGSM', nargs = "?", default = 1, type = int, help = "Number of steps of FGSM attack")
	parser.add_argument('--alphaFGSM', nargs = "?", default = 1/255, type = float, help = "Alpha (Step Size) of FGSM attack")
	parser.add_argument('--epsFGSM', nargs = "?", default = 0.007, type = float, help = "Epsilon (strength) of FGSM attack")
	
	parser.add_argument('--stepsIFGSM', nargs = "?", default = 20, type = int, help = "Number of steps of IFGSM attack")
	parser.add_argument('--alphaIFGSM', nargs = "?", default = 2/255, type = float, help = "Alpha (Step Size) of IFGSM attack")
	parser.add_argument('--epsIFGSM', nargs = "?", default = 8/255, type = float, help = "Epsilon (strength) of IFGSM attack")
	
	parser.add_argument('--stepsMIFGSM', nargs = "?", default = 20, type = int, help = "Number of steps of MIFGSM attack")
	parser.add_argument('--alphaMIFGSM', nargs = "?", default = 2/255, type = float, help = "Alpha (Step Size) of MIFGSM attack")
	parser.add_argument('--epsMIFGSM', nargs = "?", default = 8/255, type = float, help = "Epsilon (strength) of MIFGSM attack")
	parser.add_argument('--decayMIFGSM', nargs = "?", default = 1.0, type = float, help = "Decay factor of MIFGSM attack")
	
	parser.add_argument('--stepsMRFGSM', nargs = "?", default = 20, type = int, help = "Number of steps of MRFGSM attack")
	parser.add_argument('--alphaMRFGSM', nargs = "?", default = 2/255, type = float, help = "Alpha (Step Size) of MRFGSM attack")
	parser.add_argument('--epsMRFGSM', nargs = "?", default = 8/255, type = float, help = "Epsilon (strength) of MRFGSM attack")
	parser.add_argument('--decayMRFGSM', nargs = "?", default = 1.0, type = float, help = "Decay factor of MRFGSM attack")
	
	parser.add_argument('--stepsDMRIFGSM', nargs = "?", default = 20, type = int, help = "Number of steps of DMRIFGSM attack")
	parser.add_argument('--alphaDMRIFGSM', nargs = "?", default = 2/255, type = float, help = "Alpha (Step Size) of DMRIFGSM attack")
	parser.add_argument('--epsDMRIFGSM', nargs = "?", default = 8/255, type = float, help = "Epsilon (strength) of DMRIFGSM attack")
	parser.add_argument('--decayDMRIFGSM', nargs = "?", default = 1.0, type = float, help = "Decay factor of DMRIFGSM attack")
	parser.add_argument('--randomStartDMRIFGSM', nargs = "?", default = 0, type = float, help = "Decay factor of DMRIFGSM attack")

	parser.add_argument('--stepsAPMRFGSM', nargs = "?", default = 20, type = int, help = "Number of steps of APMRFGSM attack")
	parser.add_argument('--alphaAPMRFGSM', nargs = "?", default = 2/255, type = float, help = "Alpha (Step Size) of APMRFGSM attack")
	parser.add_argument('--epsAPMRFGSM', nargs = "?", default = 8/255, type = float, help = "Epsilon (strength) of APMRFGSM attack")
	parser.add_argument('--decayAPMRFGSM', nargs = "?", default = 0.99, type = float, help = "Decay factor of APMRFGSM attack")
	parser.add_argument('--decay2APMRFGSM', nargs = "?", default = 0.999, type = float, help = "Decay factor of APMRFGSM attack")

	parser.add_argument('--stepsDIFGSM', nargs = "?", default = 20, type = int, help = "Number of steps of DIFGSM attack")
	parser.add_argument('--alphaDIFGSM', nargs = "?", default = 2/255, type = float, help = "Alpha (Step Size) of DIFGSM attack")
	parser.add_argument('--epsDIFGSM', nargs = "?", default = 8/255, type = float, help = "Epsilon (strength) of DIFGSM attack")
	
	parser.add_argument('--stepsPGD', nargs = "?", default = 20, type = int, help = "Number of steps of PGD attack")
	parser.add_argument('--alphaPGD', nargs = "?", default = 2/255, type = float, help = "Alpha (Step Size) of PGD attack")
	parser.add_argument('--epsPGD', nargs = "?", default = 8/255, type = float, help = "Epsilon (strength) of PGD attack")

	parser.add_argument('--epsAutoAttack', nargs = "?", default = 8/255, type = float, help = "Epsilon (strength) of Auto attack")
	parser.add_argument('--stepsAutoAttack', nargs = "?", default = 100, type = float, help = "Number of steps of Auto attack")
	
	parser.add_argument('--stepsCW', nargs = "?", default = 1000, type = int, help = "Number of steps of CW attack")

	parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')

	parser.add_argument(
	    '--env-config',
	    default='configDefended.json',
	    metavar='EC',
	    help='environment to crop and resize info (default: config.json)')
	parser.add_argument(
	    '--load-path',
	    default='trained_models/PongNoFrameskip-v4_robust.pt',
	    metavar='LMD',
	    help='path to trained model file')
	parser.add_argument(
	    '--gpu-id',
	    type=int,
	    default=-1,
	    help='GPU to use [-1 CPU only] (default: -1)')
	parser.add_argument(
	    '--skip-rate',
	    type=int,
	    default=4,
	    metavar='SR',
	    help='frame skip rate (default: 4)')
	parser.add_argument(
	    '--fgsm-video',
	    type=float,
	    default=None,
	    metavar='FV',
	    help='whether to to produce a video of the agent performing under FGSM attack with given epsilon')
	parser.add_argument(
	    '--pgd-video',
	    type=float,
	    default=None,
	    metavar='PV',
	    help='whether to to produce a video of the agent performing under PGD attack with given epsilon')
	parser.add_argument('--video',
	                    dest='video',
	                    action='store_true',
	                    help = 'saves a video of standard eval run of model')
	parser.add_argument('--fgsm',
	                    dest='fgsm',
	                    action='store_true',
	                    help = 'evaluate against fast gradient sign attack')
	parser.add_argument('--pgd',
	                   dest='pgd',
	                   action='store_true',
	                   help='evaluate against projected gradient descent attack')
	parser.add_argument('--gwc',
	                   dest='gwc',
	                   action='store_true',
	                   help='whether to evaluate worst possible(greedy) outcome under any epsilon bounded attack')
	parser.add_argument('--action-pert',
	                   dest='action_pert',
	                   action='store_true',
	                   help='whether to evaluate performance under action perturbations')
	parser.add_argument('--acr',
	                   dest='acr',
	                   action='store_true',
	                   help='whether to evaluate the action certification rate of an agent')
	parser.add_argument('--nominal',
	                   dest='nominal',
	                   action='store_true',
	                   help='evaluate the agents nominal performance without any adversaries')

	parser.set_defaults(video=False, fgsm=False, pgd=False, gwc=False, action_pert=False, acr=False)


	return parser