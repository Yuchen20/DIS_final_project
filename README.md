# DIS_final_project



# set up 
To download the required liberaies, one can do:
```bash
python -m venv venv
source venv/bin/activate
pip install uv
uv pip install -r requirements.txt   # 100x faster pip install

# make as a kernel
python -m ipykernel install --user --name=dissertation_venv
```

## Dataset Download

This project utilizes the public cpg0000 dataset from the Cell Painting Gallery, which is hosted on AWS S3. To download the data, we provide a script, `Utils/download_data.sh`, that automates the download process; please modify the `LOCAL_ROOT` variable within this script to set your desired local storage path before executing it. 

```bash
pip install awscli                  # To allow access to AWS S3
chmod +x Utils/download_data.sh     # To make the script executable
./Utils/download_data.sh            # Start the download
```

Execution of the script will download data for the following 10 plates: `BR00116991`, `BR00116992`, `BR00116995`, `BR00117024`, `BR00116993`, `BR00117025`, `BR00116994`, `BR00117026`, `BR00118041`, and `BR00118045`. The downloaded data will be organized according to the directory structure detailed below:


```bash
├───data
│   ├───BR00116991
│   │   ├───Images
│   │   │   ├───r01c01f01p01-ch1sk1fk1fl1.tiff
│   │   │   ├───r01c01f01p01-ch2sk1fk1fl1.tiff
│   │   │   ...
│   │   │   ├───r01c01f01p01-ch8sk1fk1fl1.tiff
│   │   │   ...
│   ├───BR00116992
...
```

A dicitonary for channel to the BF, IF Images:
These are given from ChatGPT - [chat](https://chatgpt.com/share/6818db5c-1c9c-8011-a6a9-00e78cbaf0ce)
| Channel  | Dye / Filter              | Biological target                                  | Notes                                                  | Citation                 |
| -------- | ------------------------- | -------------------------------------------------- | ------------------------------------------------------ | ------------------------ |
| **ch01** | Alexa 647                 | Mitochondria                                       | MitoTracker Deep Red                                   | ([PubMed][1])            |
| **ch02** | Alexa 568                 | AGP (Actin cytoskeleton + Golgi + plasma membrane) | Phalloidin (actin) + WGA (Golgi/PM) share this channel | ([PubMed][1])            |
| **ch03** | Alexa 488 (long pass)     | Endoplasmic reticulum (ER)                         | Concanavalin A/Alexa 488 conjugate                     | ([Molecular Devices][2]) |
| **ch04** | Alexa 488                 | RNA (nucleoli + cytoplasmic RNA)                   | SYTO 14 green fluorescent nucleic acid stain           | ([Molecular Devices][2]) |
| **ch05** | Hoechst 33342             | DNA (nucleus)                                      | Hoechst nuclear stain                                  | ([PubMed][1])            |
| **ch06** | Brightfield (z – 5 µm)    | Reference BF (below focal plane)                   | BFLow                                                  | —                        |
| **ch07** | Brightfield (focal plane) | Primary BF (reference focal plane)                 | BF primary                                             | —                        |
| **ch08** | Brightfield (z + 5 µm)    | Reference BF (above focal plane)                   | BFHigh                                                 | —                        |

[1]: https://pubmed.ncbi.nlm.nih.gov/27560178/ "Cell Painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes - PubMed"
[2]: https://www.moleculardevices.com/applications/cell-imaging/cell-painting?utm_source=chatgpt.com "Cell painting, Phenotypic and Morphological Profiling"

The channel are been splitted into 3 groups:
<!-- Mito,AGP,DNA; ii) RNA,ER,DNA; and iii) DNA, nuclei
segmentation, cell segmentation -->
- Mito (ch01), AGP (ch02), DNA (ch05)
- RNA (ch04), ER (ch03), DNA (ch05)
- DNA (ch05), nuclei segmentation, cell segmentation