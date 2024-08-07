import numpy as np
from scipy.special import expit
from copy import deepcopy
from tqdm.notebook import tqdm


def get_all_param():
    allpar = {}
    allpar["mmodulTCN"] = 4.
    allpar["mmodulTRN"] = -3.
    allpar["vmodulTCN"] = 0.
    allpar["vmodulTRN"] = 0.
    allpar["tmsstim"] = False
    allpar["dt"] = 0.0001
    allpar["tfin"] = 105.
    allpar["ttms"] = allpar["tfin"]+1
    allpar["tregime"] = 30.
    allpar["tspan0"] = allpar["tregime"]
    allpar["tspanf"] = allpar["tfin"]
    allpar["mecc"] = 20.
    allpar["minh"] = 0.
    allpar["vecc"] = 0.5
    allpar["vinh"] = 0.
    allpar["wmsenseTCN"] = 'gauss'  # Modulating type (cost, exp, rect, gauss)
    allpar["wcsenseTCN"] = 'sine'  # Carrier type (cost, sine, spikes, rect)
    allpar["AsenseTCN"] = 0.  # Modulating amplitude
    allpar["ssenseTCN"] = 0.25  # Modulating dimension
    allpar["tsenseTCN"] = 40.5  # Modulating temporal position
    allpar["fsenseTCN"] = 5.  # Modulating frequency
    allpar["dsenseTCN"] = 0.2  # Modulating duty-cycle
    allpar["yptms"] = -10.
    allpar["WTCNTRN"] = -4.3
    allpar["WTCNp"] = 1.
    allpar["WTRNTCN"] = 3.
    allpar["WTRNTRN"] = -0.3
    allpar["WTRNp"] = 1.5
    allpar["WpTCN"] = 3.
    allpar["Wpp"] = 15.
    allpar["tauWpp"] = 10.
    allpar["WfTCN"] = 0.3

    dt = allpar["dt"]
    allpar["GtA"] = np.ones(int((allpar["tfin"]-allpar["tregime"])/dt+1))*1.12
    allpar["GtB"] = np.ones_like(allpar["GtA"])*0.01

    # Modulatory inputs
    t = np.array([0, 30, 45, 60, 75, 90, 105])
    modulTCN = np.array([4.5, 4.5,   4,     2,     1.5,   -1, -1])
    modulTRN = np.array([-5.5, -5.5,  -5.5,  -5,    -5,    -3, -3])
    mecc = np.array([130, 130,   50,    40,    40,    20, 20])

    new_t = np.linspace(0, allpar["tfin"],
                        int(allpar["tfin"]/dt)+1, endpoint=False)
    allpar["modulTCN"] = np.interp(new_t, t, modulTCN)
    allpar["modulTRN"] = np.interp(new_t, t, modulTRN)
    allpar["modulP"] = np.interp(new_t, t, mecc)
    rands = np.random.randn(*allpar["modulTCN"].shape)
    allpar["ecc"] = allpar["modulP"] + np.sqrt(allpar["vecc"]/dt)*rands

    allpar["senseTCN"] = np.zeros(allpar["modulTCN"].shape)
    allpar["inh"] = np.zeros(allpar["modulTCN"].shape)
    allpar["t"] = new_t

    return allpar


def thal_ctx_simulator(allpar_arg):

    allpar = deepcopy(allpar_arg)

    # Paramreters
    # Synaptic poles (rad/s)
    # ... thalamus
    wtG = 83.
    wtA = 65.
    wtB = 11.

    # ... cortex
    wcE = 75.
    wcS = 30.
    wcF = 75.

    # Calcium inactivation poles (rad/s)
    wCa1 = 10.
    wCa2 = 20.

    # Synaptic Gains (mV)
    # ... thalamus
    GtG = 4.42
    GtA = allpar["GtA"]  # 1.1
    GtB = allpar["GtB"]  # 0.01

    # ... cortex
    GcE = 5.17
    GcS = 4.45
    GcF = 57.1

    # Calcium inactivation gain
    GCa = wCa1 * wCa2

    # Number of synaptic contacts between cortical populations
    # high beta 18-19 Hz
    Cep = 1.
    Cpe = 2.
    Csp = 0.5
    Cps = 0.5
    Cfp = 3.
    Cfs = 0.5
    Cpf = 3.5
    Cff = 1.3

    # Sigmoids parameters
    # 1-tonic (thalamus)
    sigmaf = -1.
    thetaf = 5.

    # 2-calcium inactivation (thalamus)
    sigmaCa = 1.
    thetaCa = -3.

    # 3-burst (thalamus)
    sigmam = -0.1
    thetam = 0.

    # 4-GABAB (thalamus)
    sigmaB = -30.
    thetaB = 200.

    # 5-cortical
    sigmac = -1/0.56
    thetac = 6.

    Gzt = 50.  # Tonic maximum tonic firing rate (Hz) (thalamus)
    Gzb = 800.  # Tonic maximum burst firing rate (Hz) (thalamus)
    Gzc = 50.  # Tonic maximum tonic firing rate (Hz) (cortex)

    dt = allpar["dt"]  # Integration step (s)
    tfin = allpar["tfin"]  # Duration (s)
    N = int(np.round(tfin/dt))  # Number of integration steps
    ds = 0.001  # Sampling step (s)
    rd = int(np.round(ds/dt))
    tregime = allpar["tregime"]
    tout = np.arange(tregime, tfin, ds)
    Nout = len(tout)  # Number of output samples
    DW = 0.001  # Long-range connections finite-delay (s)
    NDW = int(np.round(DW/dt))  # Number of delay samples
    saveoffset = int(np.ceil(tregime/ds))

    # Generates all the random numbers (important when need to set random seed
    # and simulation duration can change)
    randomnumbers = np.random.randn(4, N)

    # Thalamic modules inputs
    if 'modulTCN' in allpar and 'modulTRN' in allpar and 'senseTCN' in allpar:
        modulTCN = wtG/GtG*allpar["modulTCN"]
        modulTRN = wtG/GtG*allpar["modulTRN"]
        senseTCN = allpar["senseTCN"]
        del allpar['modulTCN']
        del allpar['modulTRN']
        del allpar['senseTCN']
    else:
        mmodulTCN = allpar["mmodulTCN"]
        mmodulTRN = allpar["mmodulTRN"]
        vmodulTCN = allpar["vmodulTCN"]
        vmodulTRN = allpar["vmodulTRN"]

        wmsenseTCN = allpar["wmsenseTCN"]
        wcsenseTCN = allpar["wcsenseTCN"]
        AsenseTCN = allpar["AsenseTCN"]
        ssenseTCN = allpar["ssenseTCN"]
        tsenseTCN = allpar["tsenseTCN"]
        fsenseTCN = allpar["fsenseTCN"]

        modulTCN = wtG/GtG*mmodulTCN+np.sqrt(vmodulTCN/dt)*randomnumbers[0, :]
        modulTRN = wtG/GtG*mmodulTRN+np.sqrt(vmodulTRN/dt)*randomnumbers[1, :]
        tinput = np.arange(0, tfin, dt)
        if wmsenseTCN == 'cost':
            senseTCN = AsenseTCN
        elif wmsenseTCN == 'gauss':
            senseTCN = AsenseTCN*np.exp(-0.5*((tinput-tsenseTCN)/ssenseTCN)**2)

        if wcsenseTCN == 'cost':
            senseTCN = senseTCN*np.ones(*tinput.shape)
        elif wmsenseTCN == 'sine':
            senseTCN = 0.5*senseTCN*(np.cos(2*np.pi*fsenseTCN*tinput)+1)

    # Cortical modules input
    if "ecc" in allpar and 'inh' in allpar:
        ecc = allpar["ecc"]
        inh = allpar["inh"]
        del allpar["ecc"]
        del allpar["inh"]
    else:
        mecc = allpar["mecc"]
        minh = allpar["minh"]
        vecc = allpar["vecc"]
        vinh = allpar["vinh"]

        ecc = mecc + np.sqrt(vecc/dt)*randomnumbers[2, :]
        inh = minh + np.sqrt(vinh/dt)*randomnumbers[3, :]

    # TMS stimulus
    tmsstim = allpar["tmsstim"]
    yptms = allpar["yptms"]
    ttms = allpar["ttms"]

    # State variables
    zCatTCN, zCatTRN = 0, 0     # Calcium inactivation (thalamus)
    zCa1tTCN, zCa1tTRN = 0, 0   # ... derivatives

    # Pre-synaptic membrane potentials (mV)
    ytG, ytA, ytB = 0, 0, 0               # (thalamus)
    yp, ye, ys, yf = 0, 0, 0, 0           # (cortex)
    yWTCN, yWTRN, yWp, yWf = 0, 0, 0, 0   # (connection)

    # ... derivatives
    xtG, xtA, xtB = 0, 0, 0               # (thalamus)
    xp, xe, xs, xf = 0, 0, 0, 0           # (cortex)
    xWTCN, xWTRN, xWp, xWf = 0, 0, 0, 0   # (connection)

    # Post-synaptic membrane potentials
    """
    thalamus                        : vTCN, vTRN
    cortex                          : vp, vf
    Firing rates burst (thalamus)   : zbTCN, zbTRN
    Firing rates calcium (thalamus) : zCa2TRN, zCaTRN, zmTRN
    Firing rates tonic (thalamus)   : zfTCN, zfTRN
    Firing rates cortical           : zp, ze, zs, zf
    """
    postsyn_keys = ['vTCN', 'vTRN', 'vp', 'vf', 'zbTCN', 'zbTRN',
                    'zCa2TRN', 'zCaTRN', 'zmTRN', 'zfTCN', 'zfTRN',
                    'zp', 'ze', 'zs', 'zf']
    out_signals = {key: np.zeros(Nout) for key in postsyn_keys}

    # ... long range connections
    zWTCN = np.zeros(NDW)
    zWp = np.zeros(NDW)

    # Connectivity
    WTCNTRN = allpar["WTCNTRN"]
    WTCNp = allpar["WTCNp"]
    WTRNTCN = allpar["WTRNTCN"]
    WTRNTRN = allpar["WTRNTRN"]
    WTRNp = allpar["WTRNp"]
    WpTCN = allpar["WpTCN"]
    Wpp = allpar["Wpp"]
    WfTCN = allpar["WfTCN"]

    # cortical disfacilitation
    Wppmax = Wpp
    gamma_Wpp = dt/(allpar["tauWpp"]/4.5)

    dvars = 0
    d_regime = int(tregime/dt)
    for d, t in tqdm(tuple(enumerate(np.arange(0, tfin, dt)))):
        in_regime = t >= tregime
        dW = d % NDW
        if in_regime:
            dvars = d-d_regime

        # Potentials
        vtTCN = WTCNTRN*(ytA+ytB)+yWTCN
        vtTRN = WTRNTCN*ytG+WTRNTRN*(ytA+ytB)+yWTRN

        vpt = Cpe*ye-Cps*ys-Cpf*yf+yWp
        vet = Cep*yp
        vst = Csp*yp
        vft = Cfp*yp-Cfs*ys-Cff*yf+yWf

        # 1/(1+exp(-x)) = expit(x)
        # Calcium inactivation
        zCa2tTCN = expit((thetaCa-vtTCN)/sigmaCa)
        zCa2tTRN = expit((thetaCa-vtTRN)/sigmaCa)

        dzCatTCN = zCa1tTCN
        dzCatTRN = zCa1tTRN

        dzCa1tTCN = GCa*zCa2tTCN-(wCa1+wCa2)*zCa1tTCN-GCa*zCatTCN
        dzCa1tTRN = GCa*zCa2tTRN-(wCa1+wCa2)*zCa1tTRN-GCa*zCatTRN

        # Burst firing
        zmtTCN = expit((thetam-vtTCN)/sigmam)
        zmtTRN = expit((thetam-vtTRN)/sigmam)

        zbtTCN = zCatTCN*zmtTCN
        zbtTRN = zCatTRN*zmtTRN

        # Tonic firing
        zftTCN = expit((thetaf-vtTCN)/sigmaf)
        zftTRN = expit((thetaf-vtTRN)/sigmaf)

        # Firing rates
        ztG = zbtTCN*Gzb+(1-zbtTCN)*Gzt*zftTCN
        ztA = zbtTRN*Gzb+(1-zbtTRN)*Gzt*zftTRN
        ztB = ztA*expit((thetaB-ztA)/sigmaB)

        zpt = Gzc*expit((thetac-vpt)/sigmac)
        zet = Gzc*expit((thetac-vet)/sigmac)
        zst = Gzc*expit((thetac-vst)/sigmac)
        zft = Gzc*expit((thetac-vft)/sigmac)

        dxtG = wtG*(GtG*ztG-2*xtG-wtG*ytG)

        dxtA = wtA*(GtA[dvars]*ztA-2*xtA-wtA*ytA)
        dxtB = wtB*(GtB[dvars]*ztB-2*xtB-wtB*ytB)

        dxp = wcE*(GcE*zpt-2*xp-wcE*yp)
        dxe = wcE*(GcE*zet-2*xe-wcE*ye)
        dxs = wcS*(GcS*zst-2*xs-wcS*ys)
        dxf = wcF*(GcF*zft-2*xf-wcF*yf)

        inputTCN = senseTCN[d]+modulTCN[d]+WTCNp*zWp[dW]
        inputTRN = modulTRN[d]+WTRNp*zWp[dW]

        inputp = ecc[d]+WpTCN*zWTCN[dW]+Wpp*zWp[dW]
        inputf = inh[d]+WfTCN*zWTCN[dW]

        dxWTCN = wtG*(GtG*inputTCN-2*xWTCN-wtG*yWTCN)
        dxWTRN = wtG*(GtG*inputTRN-2*xWTRN-wtG*yWTRN)
        dxWp = wcE*(GcE*inputp-2*xWp-wcE*yWp)
        dxWf = wcE*(GcE*inputf-2*xWf-wcE*yWf)

        # State variables update
        zCatTCN += dzCatTCN*dt
        zCatTRN += dzCatTRN*dt
        zCa1tTCN += dzCa1tTCN*dt
        zCa1tTRN += dzCa1tTRN*dt

        ytG += xtG*dt  # dytG*dt
        ytA += xtA*dt  # dytA*dt
        ytB += xtB*dt  # dytB*dt

        yp += xp*dt   # dyp*dt
        ye += xe*dt   # dye*dt
        ys += xs*dt   # dys*dt
        yf += xf*dt   # dyf*dt
        yWTCN += xWTCN*dt  # dyWTCN*dt
        yWTRN += xWTRN*dt  # dyWTRN*dt
        yWp += xWp*dt  # dyWp*dt
        yWf += xWf*dt  # dyWf*dt

        xtG += dxtG*dt
        xtA += dxtA*dt
        xtB += dxtB*dt

        xp += dxp*dt
        xe += dxe*dt
        xs += dxs*dt
        xf += dxf*dt

        xWTCN += dxWTCN*dt
        xWTRN += dxWTRN*dt
        xWp += dxWp*dt
        xWf += dxWf*dt

        if in_regime:
            if not (d % rd):
                # Output variables downsampling
                i = int(d/rd-saveoffset)
                out_signals["vTCN"][i] = vtTCN
                out_signals["vTRN"][i] = vtTRN
                out_signals["zCa2TRN"][i] = zCa2tTRN
                out_signals["zCaTRN"][i] = zCatTRN
                out_signals["zmTRN"][i] = zmtTRN
                out_signals["zbTCN"][i] = zbtTCN*Gzb
                out_signals["zbTRN"][i] = zbtTRN*Gzb
                out_signals["zfTCN"][i] = (1-zbtTCN)*Gzt*zftTCN
                out_signals["zfTRN"][i] = (1-zbtTRN)*Gzt*zftTRN
                out_signals["vp"][i] = vpt
                out_signals["vf"][i] = vft
                out_signals["zp"][i] = zpt
                out_signals["ze"][i] = zet
                out_signals["zs"][i] = zst
                out_signals["zf"][i] = zft

        zWTCN[dW] = ztG
        zWp[dW] = zpt

        Wpp = Wpp*(1-gamma_Wpp)+gamma_Wpp*Wppmax*(1-zpt/Gzc)

        # TMS stimulus
        if tmsstim:
            if np.abs(t-ttms) < 0.5*dt:
                yp += yptms
                xp, xe, xs, xf = 0, 0, 0, 0

                vpt = Cpe*ye-Cps*ys-Cpf*yf+yWp
                zpt = Gzc/(1+np.exp((vpt-thetac)/sigmac))

                out_signals["vp"][int(d/rd-saveoffset)] = vpt
                out_signals["zp"][int(d/rd-saveoffset)] = zpt
                zWp[dW] = zpt

    return out_signals
