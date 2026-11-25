package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Dataset struct {
	Name      string
	Task      string
	PageURL   string
	DirectURL string // Empty if requires Kaggle CLI, Python, or manual agreement
	Filename  string
	Note      string
}

var datasets = []Dataset{
	// --- 5 Current Datasets ---
	{
		Name:      "HIGGS",
		Task:      "Binary Classification (11M rows, 28 features)",
		PageURL:   "https://archive.ics.uci.edu/ml/datasets/HIGGS",
		DirectURL: "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
		Filename:  "HIGGS.csv.gz",
	},
	{
		Name:      "Covertype",
		Task:      "Multi-class (581K rows, 54 features)",
		PageURL:   "https://archive.ics.uci.edu/ml/datasets/covertype",
		DirectURL: "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
		Filename:  "covtype.data.gz",
	},
	{
		Name:      "Adult Income",
		Task:      "Binary Classification (48.8K rows, 14 features)",
		PageURL:   "https://archive.ics.uci.edu/ml/datasets/adult",
		DirectURL: "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
		Filename:  "adult.data",
	},
	{
		Name:    "CreditCard Fraud",
		Task:    "Binary Classification (284.8K rows, 30 features)",
		PageURL: "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
		Note:    "Requires Kaggle CLI: kaggle datasets download -d mlg-ulb/creditcardfraud",
	},
	{
		Name:    "California Housing",
		Task:    "Regression (20.6K rows, 8 features)",
		PageURL: "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html",
		Note:    "Fetch via Python: sklearn.datasets.fetch_california_housing(as_frame=True)",
	},

	// --- 6 Recommended Additions ---
	{
		Name:      "YearPredictionMSD",
		Task:      "Regression (515K rows, 90 features)",
		PageURL:   "https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd",
		DirectURL: "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
		Filename:  "YearPredictionMSD.txt.zip",
	},
	{
		Name:    "Epsilon",
		Task:    "Binary Classification (500K rows, 2,000 features)",
		PageURL: "https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=Epsilon",
		Note:    "Fetch via OpenML API or Python: sklearn.datasets.fetch_openml(data_id=45575)",
	},
	{
		Name:      "SUSY",
		Task:      "Binary Classification (5M rows, 18 features)",
		PageURL:   "https://archive.ics.uci.edu/ml/datasets/SUSY",
		DirectURL: "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz",
		Filename:  "SUSY.csv.gz",
	},
	{
		Name:      "KDD Cup 1999",
		Task:      "Multi-class (4M+ rows, 41 features)",
		PageURL:   "https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data",
		DirectURL: "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz",
		Filename:  "kddcup.data.gz",
	},
	{
		Name:      "Bank Marketing",
		Task:      "Binary Classification (45.2K rows, 17 features)",
		PageURL:   "https://archive.ics.uci.edu/dataset/222/bank+marketing",
		DirectURL: "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
		Filename:  "bank-additional.zip",
	},
	{
		Name:      "Diabetes 130-US Hospitals",
		Task:      "Multi-class / Binary (101.7K rows, 47 features)",
		PageURL:   "https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008",
		DirectURL: "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
		Filename:  "dataset_diabetes.zip",
	},

	// --- 12th Benchmark Slot ---
	{
		Name:    "MSLR-WEB10K",
		Task:    "Learning-to-Rank (1.2M rows, 136 features)",
		PageURL: "https://www.microsoft.com/en-us/research/project/mslr/",
		Note:    "Download manually from Microsoft Research portal (license agreement required)",
	},
}

func downloadFile(client *http.Client, outDir string, d Dataset) error {
	outPath := filepath.Join(outDir, d.Filename)
	if _, err := os.Stat(outPath); err == nil {
		fmt.Printf("[SKIP] %s already exists at %s\n", d.Name, outPath)
		return nil
	}

	fmt.Printf("[START] Downloading %s...\n", d.Name)
	resp, err := client.Get(d.DirectURL)
	if err != nil {
		return fmt.Errorf("%s: request failed: %w", d.Name, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: HTTP %d (%s)", d.Name, resp.StatusCode, d.DirectURL)
	}

	out, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("%s: create file failed: %w", d.Name, err)
	}
	defer out.Close()

	n, err := io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("%s: write failed: %w", d.Name, err)
	}

	fmt.Printf("[DONE] %s -> %s (%.2f MB)\n", d.Name, outPath, float64(n)/(1024*1024))
	return nil
}

func main() {
	download := flag.Bool("download", false, "Download all datasets with direct URLs")
	outDir := flag.String("out", "./datasets", "Output directory for downloaded datasets")
	workers := flag.Int("workers", 8, "Max concurrent downloads")
	flag.Parse()

	if !*download {
		for i, d := range datasets {
			fmt.Printf("%2d. %s — %s\n", i+1, d.Name, d.Task)
			fmt.Printf("    Page:   %s\n", d.PageURL)
			if d.DirectURL != "" {
				fmt.Printf("    Direct: %s\n", d.DirectURL)
			}
			if d.Note != "" {
				fmt.Printf("    Note:   %s\n", d.Note)
			}
			fmt.Println()
		}
		fmt.Println("Run with -download to fetch direct-downloadable archives.")
		return
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create directory %s: %v\n", *outDir, err)
		os.Exit(1)
	}

	client := &http.Client{Timeout: 30 * time.Minute}
	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup

	for _, d := range datasets {
		if d.DirectURL == "" {
			fmt.Printf("[MANUAL] %s: %s (%s)\n", d.Name, d.Note, d.PageURL)
			continue
		}

		wg.Add(1)
		go func(ds Dataset) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			if err := downloadFile(client, *outDir, ds); err != nil {
				fmt.Fprintf(os.Stderr, "[ERR] %v\n", err)
			}
		}(d)
	}

	wg.Wait()
}