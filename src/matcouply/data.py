import json
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
import tensorly as tl
from scipy.io import loadmat
from tqdm import tqdm

from matcouply.coupled_matrices import CoupledMatrixFactorization

DATASET_PARENT = Path(__file__).parent / "datasets"
DOWNLOADED_PARENT = DATASET_PARENT / "downloads"


def get_simple_simulated_data(noise_level=0.2, random_state=1):
    r"""Generate a simple simulated dataset with shifting unimodal :math:`\mathbf{B}^{(i)}` matrices.

    The entries in :math:`\mathbf{A}` (or :math:`\mathbf{D}^{(i)}`-matrices) are uniformly distributed
    between 0.1 and 1.1. This is done to ensure that there is signal from all components in all matrices.

    The component vectors in the :math:`\mathbf{B}^{(i)}` matrices are Gaussian probability density
    functions that shift one entry for each matrix. This means that they are non-negative, unimodal
    and satisfy the PARAFAC2 constraint.

    The entries in :math:`\mathbf{C}` are truncated normal distributed, and are therefore sparse.

    The dataset is generated by constructing the matrices represented by the decomposition and adding
    noise according to

    .. math::

        \mathbf{M}_\text{noisy}^{(i)} =
        \mathbf{M}^{(i)} + \eta \frac{\|\mathbf{\mathcal{M}}\|}{\|\mathbf{\mathcal{N}}\|} \mathbf{N}^{(i)},


    where :math:`\eta` is the noise level, :math:`\mathbf{M}^{(i)}` is the :math:`i`-th matrix represented
    by the simulated factorization, :math:`\mathbf{\mathcal{M}}` is the tensor obtained by stacking all the
    :math:`\mathbf{M}^{(i)}`-matrices, :math:`n^{(i)}_{jk} \sim \mathcal{N}(0, 1)` and :math:`\mathbf{N}^{(i)}`
    and :math:`\mathbf{\mathcal{N}}` and is the matrix and tensor with elements given by :math:`n^{(i)}_{jk}`, respectively.

    Parameters
    ----------
    noise_level : float
        Strength of noise added to the matrices
    random_state : None, int or valid tensorly random state

    Returns
    -------
    list of matrices
        The noisy matrices
    CoupledMatrixFactorization
        The factorization that underlies the simulated data matrices
    """
    rank = 3
    I, J, K = 15, 50, 20

    rng = tl.check_random_state(random_state)

    # Uniformly distributed A
    A = rng.uniform(size=(I, rank)) + 0.1  # Add 0.1 to ensure that there is signal for all components for all slices
    A = tl.tensor(tl.tensor(A))

    # Generate B-matrix as shifting normal distributions
    t = np.linspace(-10, 10, J)
    B_blueprint = np.stack([stats.norm.pdf(t, loc=-5), stats.norm.pdf(t, loc=0), stats.norm.pdf(t, loc=2)], axis=-1)
    B_is = [np.roll(B_blueprint, i, axis=0) for i in range(I)]  # Cyclically permute to get changing B_i matrices
    B_is = [tl.tensor(B_i) for B_i in B_is]

    # Truncated normal distributed C
    C = tl.tensor(rng.standard_normal(size=(K, rank)))
    C[C < 0] = 0
    C = tl.tensor(C)

    # Construct decomposition and data matrix
    cmf = CoupledMatrixFactorization((None, (A, B_is, C)))
    matrices = cmf.to_matrices()

    # Add noise
    noise = [tl.tensor(rng.standard_normal(size=M.shape)) for M in matrices]
    scale_factor = tl.norm(tl.stack(matrices)) / tl.norm(tl.stack(noise))
    matrices = [M + noise_level * scale_factor * N for M, N in zip(matrices, noise)]
    return matrices, cmf


def get_bike_data():
    r"""Get bike sharing data from three major Norwegian cities

    This dataset contains three matrices with bike sharing data from Oslo, Bergen and Trondheim,
    :math:`\mathbf{X}^{(\text{Oslo})}, \mathbf{X}^{(\text{Bergen})}` and :math:`\mathbf{X}^{(\text{Trondheim})}`.
    Each row of these data matrices represent a station, and each column of the data matrices
    represent an hour in 2021. The matrix element :math:`x^{(\text{Oslo})}_{jk}` is the number of trips
    that ended in station :math:`j` in Oslo during hour :math:`k`.

    The data was obtained using the open API of

     * Oslo Bysykkel: https://oslobysykkel.no/en/open-data
     * Bergen Bysykkel: https://bergenbysykkel.no/en/open-data
     * Trondheim Bysykkel: https://trondheimbysykkel.no/en/open-data

    on the 23rd of November 2021.

    The dataset is cleaned so it only contains for the dates in 2021 when bike sharing was open in all three
    cities (2021-04-07 - 2021-11-23).

    Returns
    -------
    dict
        Dictionary mapping the city name with a data frame that contain bike sharing data from that city.
        There is also an additional ``"station_metadata"``-key, which maps to a data frame with additional
        station metadata. This metadata is useful for interpreting the extracted components.

    .. note::

        The original bike sharing data is released under a NLOD lisence (https://data.norge.no/nlod/en/2.0/).
    """
    with ZipFile(DATASET_PARENT / "bike.zip") as data_zip:
        with data_zip.open("bike.json", "r") as f:
            bike_data = json.load(f)

    datasets = {key: pd.DataFrame(value) for key, value in bike_data["dataset"].items()}
    time = [datetime(2021, 1, 1) + timedelta(hours=int(h)) for h in datasets["oslo"].columns]
    time = pd.to_datetime(time).tz_localize("UTC").tz_convert("CET")

    out = {}
    for city in ["oslo", "trondheim", "bergen"]:
        df = datasets[city]
        df.columns = time
        df.columns.name = "Time of arrival"
        df.index.name = "Arrival station ID"
        df.index = df.index.astype(int)
        out[city] = df

    out["station_metadata"] = datasets["station_metadata"]
    out["station_metadata"].index.name = "Arrival station ID"
    out["station_metadata"].index = out["station_metadata"].index.astype(int)

    return out


def get_semiconductor_etch_raw_data(download_data=True, save_data=True):
    """Load semiconductor etch data from :cite:p:`wise1999comparison`.

    If the dataset is already downloaded on your computer, then the local files will be
    loaded. Otherwise, they will be downloaded. By default, the files are downloaded from
    https://eigenvector.com/data/Etch.

    Parameters
    ----------
    download_data : bool
        If ``False``, then an error will be raised if the data is not
        already downloaded.
    save_data : bool
        if ``True``, then the data will be stored locally to avoid having to download
        multiple times.

    Returns
    -------
    dict
        Dictionary where the keys are the dataset names and the values are the contents
        of the MATLAB files.
    """
    data_urls = {
        "MACHINE_Data.mat": "http://eigenvector.com/data/Etch/MACHINE_Data.mat",
        "OES_DATA.mat": "http://eigenvector.com/data/Etch/OES_DATA.mat",
        "RFM_DATA.mat": "http://eigenvector.com/data/Etch/RFM_DATA.mat",
    }
    data_raw_mat = {}

    print("Loading semiconductor etch data from Wise et al. (1999) - J. Chemom. 13(3‐4), pp.379-396.")
    print("The data is available at: http://eigenvector.com/data/Etch/")
    for file, url in tqdm(data_urls.items()):
        file_path = DOWNLOADED_PARENT / file
        if file_path.exists():
            data_raw_mat[file] = loadmat(file_path)
        elif download_data:
            request = requests.get(url)
            if request.status_code != 200:
                raise RuntimeError(f"Cannot download file {url} - Response: {request.status_code} {request.reason}")

            if save_data:
                DOWNLOADED_PARENT.mkdir(exist_ok=True, parents=True)
                with open(file_path, "wb") as f:
                    f.write(request.content)

            data_raw_mat[file] = loadmat(BytesIO(request.content))
        else:
            raise RuntimeError("The semiconductor etch data is not yet downloaded, and ``download_data=False``.")
    return data_raw_mat


def get_semiconductor_etch_machine_data(download_data=True, save_data=True):
    """Load machine measurements from the semiconductor etch dataset from :cite:p:`wise1999comparison`.

    This function will load the semiconductor etch machine data and prepare it for analysis.

    If the dataset is already downloaded on your computer, then the local files will be
    loaded. Otherwise, they will be downloaded. By default, the files are downloaded from
    https://eigenvector.com/data/Etch.

    Parameters
    ----------
    download_data : bool
        If ``False``, then an error will be raised if the data is not
        already downloaded.
    save_data : bool
        if ``True``, then the data will be stored locally to avoid having to download
        multiple times.

    Returns
    -------
    dict
        Dictionary where the keys are the dataset names and the values are the contents
        of the MATLAB files.
    """
    # Get raw MATLAB data and parse into Python dict
    data = get_semiconductor_etch_raw_data(download_data=download_data, save_data=save_data)["MACHINE_Data.mat"][
        "LAMDATA"
    ]
    data = {key: data[key].squeeze().item().squeeze() for key in data.dtype.fields}

    # Format data as xarray dataset
    varnames = data["variables"][2:]

    # Get the training data
    train_names = [name.split(".")[0][1:] for name in data["calib_names"]]
    train_data = {
        name: pd.DataFrame(Xi[:-1, 2:], columns=varnames)  # Slice away last row since it "belongs" to the next sample
        for name, Xi in zip(train_names, data["calibration"])
    }
    train_metadata = {}
    for i, name in enumerate(list(train_data)):
        train_data[name].index.name = "Time point"
        train_data[name].columns.name = "Measurement"

        metadata = pd.DataFrame(data["calibration"][i][:-1, [0, 1]], columns=["Time", "Step number"])
        metadata["Experiment"] = int(name[:2])
        metadata["Sample"] = int(name[2:])
        metadata.index.name = "Time point"
        metadata.columns.name = "Metadata"
        train_metadata[name] = metadata

    # Get the testing data
    test_names = [name.split(".")[0][1:] for name in data["test_names"]]
    test_data = {
        name: pd.DataFrame(Xi[:-1, 2:], columns=varnames)  # Slice away last row since it "belongs" to the next sample
        for name, Xi in zip(test_names, data["test"])
    }
    test_metadata = {}
    for i, name in enumerate(list(test_data)):
        test_data[name].index.name = "Time point"
        test_data[name].columns.name = "Measurement"

        metadata = pd.DataFrame(data["test"][i][:-1, [0, 1]], columns=["Time", "Step number"])
        metadata["Experiment"] = int(name[:2])
        metadata["Sample"] = int(name[2:])
        metadata["Fault name"] = data["fault_names"][i]
        metadata.index.name = "Time point"
        metadata.columns.name = "Metadata"
        test_metadata[name] = metadata

    return train_data, train_metadata, test_data, test_metadata
