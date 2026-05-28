Last login: Wed May 27 17:39:38 on ttys000
alex@MacBook-Pro-2 ~ % cd prog/ongh/ddpm_rf 
alex@MacBook-Pro-2 ddpm_rf % ls
README.md		celeb128		ddmpwork_v2.py		jpg			out_images		rf1step_v2.py		upscaler_version5.py
archive			checkpoints		evualate_v2.py		logs			prep128und64.py		src			upscaler_version6.py
backup_checkpoints	data			fake064			main.py			pyproject.toml		upscaler_version2.py	vortrag
celeb064		ddmp_v2.py		hard_rf			out			real_images		upscaler_version3.py
alex@MacBook-Pro-2 ddpm_rf % cd checkpoints 
alex@MacBook-Pro-2 checkpoints % ls
ddpm_final.pth		ddpm_halfway.pth
alex@MacBook-Pro-2 checkpoints % mkdir ddmp_v2
alex@MacBook-Pro-2 checkpoints % mv * ddmp_v2 
mv: rename ddmp_v2 to ddmp_v2/ddmp_v2: Invalid argument
alex@MacBook-Pro-2 checkpoints % ls
ddmp_v2
alex@MacBook-Pro-2 checkpoints % date
Wed May 27 18:00:36 CEST 2026
alex@MacBook-Pro-2 checkpoints % date
Wed May 27 18:00:41 CEST 2026
alex@MacBook-Pro-2 checkpoints % cd ..
alex@MacBook-Pro-2 ddpm_rf % git status 
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.gitignore
	.python-version
	.spyproject/
	README.md
	archive/
	backup_checkpoints/
	celeb064/
	celeb128/
	checkpoints/
	data/
	ddmp_v2.py
	ddmpwork_v2.py
	evualate_v2.py
	fake064/
	hard_rf/
	jpg/
	logs/
	main.py
	out/
	out_images/
	prep128und64.py
	pyproject.toml
	real_images/
	rf1step_v2.py
	rf4step_v4.py
	src/
	upscaler_version2.py
	upscaler_version3.py
	upscaler_version5.py
	upscaler_version6.py
	vortrag/

nothing added to commit but untracked files present (use "git add" to track)
alex@MacBook-Pro-2 ddpm_rf % git add *.py
alex@MacBook-Pro-2 ddpm_rf %            
alex@MacBook-Pro-2 ddpm_rf % 
alex@MacBook-Pro-2 ddpm_rf % 
alex@MacBook-Pro-2 ddpm_rf % open .
alex@MacBook-Pro-2 ddpm_rf % date
Wed May 27 18:02:05 CEST 2026
alex@MacBook-Pro-2 ddpm_rf % bc -l
>>> 7200 * 2 / 60
240.00000000000000000000
>>> 72 * 2 / 60
2.40000000000000000000
>>> 72 / 60
1.20000000000000000000
>>> ^D
alex@MacBook-Pro-2 ddpm_rf % git status
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   ddmp_v2.py
	new file:   ddmpwork_v2.py
	new file:   evualate_v2.py
	new file:   main.py
	new file:   prep128und64.py
	new file:   rf1step_v2.py
	new file:   rf4step_v4.py
	new file:   upscaler_version2.py
	new file:   upscaler_version3.py
	new file:   upscaler_version5.py
	new file:   upscaler_version6.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.gitignore
	.python-version
	.spyproject/
	README.md
	archive/
	backup_checkpoints/
	celeb064/
	celeb128/
	checkpoints/
	data/
	fake064/
	hard_rf/
	jpg/
	logs/
	out/
	out_images/
	pyproject.toml
	real_images/
	src/
	vortrag/

alex@MacBook-Pro-2 ddpm_rf % grep fake *.py
upscaler_version2.py:        fake_labels = torch.zeros(B, 1, device=self.device)
upscaler_version2.py:            x_fake = self.rf.sample(B)
upscaler_version2.py:        x_fake = self.add_noise(x_fake)
upscaler_version2.py:        pred_fake = self.disc(x_fake.detach())
upscaler_version2.py:            F.binary_cross_entropy_with_logits(pred_fake, fake_labels)
upscaler_version3.py:        fake_labels = torch.zeros(B, 1, device=self.device)
upscaler_version3.py:            x_fake = self.rf.sample(B)
upscaler_version3.py:        x_fake = self.add_noise(x_fake)
upscaler_version3.py:        pred_fake = self.disc(x_fake.detach())
upscaler_version3.py:            F.binary_cross_entropy_with_logits(pred_fake, fake_labels)
upscaler_version5.py:        fake_labels = torch.zeros(B, 1, device=self.device)
upscaler_version5.py:            x_fake = self.rf.sample(B)
upscaler_version5.py:        x_fake = self.add_noise(x_fake)
upscaler_version5.py:        pred_fake = self.disc(x_fake.detach())
upscaler_version5.py:            F.binary_cross_entropy_with_logits(pred_fake, fake_labels)
upscaler_version6.py:        fake_labels = torch.zeros(B, 1, device=self.device)
upscaler_version6.py:            x_fake = self.rf.sample(B)
upscaler_version6.py:        x_fake = self.add_noise(x_fake)
upscaler_version6.py:        pred_fake = self.disc(x_fake.detach())
upscaler_version6.py:            F.binary_cross_entropy_with_logits(pred_fake, fake_labels)
alex@MacBook-Pro-2 ddpm_rf % ls
README.md		celeb128		ddmpwork_v2.py		jpg			out_images		rf1step_v2.py		upscaler_version3.py
archive			checkpoints		evualate_v2.py		logs			prep128und64.py		rf4step_v4.py		upscaler_version5.py
backup_checkpoints	data			fake064			main.py			pyproject.toml		src			upscaler_version6.py
celeb064		ddmp_v2.py		hard_rf			out			real_images		upscaler_version2.py	vortrag
alex@MacBook-Pro-2 ddpm_rf % python evualate_v2.py 
Traceback (most recent call last):
  File "/Users/alex/prog/ongh/ddpm_rf/evualate_v2.py", line 85, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
alex@MacBook-Pro-2 ddpm_rf % acvenv
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % python evualate_v2.py
Traceback (most recent call last):
  File "/Users/alex/prog/ongh/ddpm_rf/evualate_v2.py", line 87, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % pip install pandas
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pandas in /opt/local/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (2.3.3)
Requirement already satisfied: numpy>=1.26.0 in /opt/local/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from pandas) (2.4.4)
Requirement already satisfied: python-dateutil>=2.8.2 in /Users/alex/Library/Python/3.14/lib/python/site-packages (from pandas) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /opt/local/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from pandas) (2026.1.post1)
Requirement already satisfied: tzdata>=2022.7 in /Users/alex/Library/Python/3.14/lib/python/site-packages (from pandas) (2025.3)
Requirement already satisfied: six>=1.5 in /Users/alex/Library/Python/3.14/lib/python/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % python
Python 3.14.4 (main, Apr  8 2026, 18:13:21) [Clang 21.0.0 (clang-2100.0.123.102)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    import pandas
ModuleNotFoundError: No module named 'pandas'
>>> import panda
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    import panda
ModuleNotFoundError: No module named 'panda'
>>> import pandas
Traceback (most recent call last):
  File "<python-input-2>", line 1, in <module>
    import pandas
ModuleNotFoundError: No module named 'pandas'
>>> import pandas
Traceback (most recent call last):
  File "<python-input-3>", line 1, in <module>
    import pandas
ModuleNotFoundError: No module named 'pandas'
>>> quit
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % uv pip install pandas
Resolved 4 packages in 486ms
Prepared 1 package in 3.03s
Installed 1 package in 34ms
 + pandas==3.0.3
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % deactivate
alex@MacBook-Pro-2 ddpm_rf % acvenv
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % python evualate_v2.py
working on dir (saving report there)= ./out/samples_onestep35057
============================================================
EVALUATING IMAGES
============================================================

Image: img_0.png
Impact Score: 269.94
Sharpness: 653.06
Contrast: 46.08
Entropy: 2.80
Edge Density: 30.32

Image: img_1.png
Impact Score: 175.52
Sharpness: 354.63
Contrast: 44.01
Entropy: 2.66
Edge Density: 23.59

Image: img_10.png
Impact Score: 269.94
Sharpness: 653.06
Contrast: 46.08
Entropy: 2.80
Edge Density: 30.32

Image: img_11.png
Impact Score: 175.52
Sharpness: 354.63
Contrast: 44.01
Entropy: 2.66
Edge Density: 23.59

Image: img_12.png
Impact Score: 100.55
Sharpness: 170.70
Contrast: 31.41
Entropy: 1.92
Edge Density: 16.56

Image: img_13.png
Impact Score: 331.89
Sharpness: 668.96
Contrast: 80.93
Entropy: 4.89
Edge Density: 47.00

Image: img_14.png
Impact Score: 74.83
Sharpness: 137.29
Contrast: 32.38
Entropy: 1.18
Edge Density: 10.65

Image: img_15.png
Impact Score: 288.33
Sharpness: 504.56
Contrast: 77.70
Entropy: 5.21
Edge Density: 52.11

Image: img_2.png
Impact Score: 100.55
Sharpness: 170.70
Contrast: 31.41
Entropy: 1.92
Edge Density: 16.56

Image: img_3.png
Impact Score: 331.89
Sharpness: 668.96
Contrast: 80.93
Entropy: 4.89
Edge Density: 47.00

Image: img_4.png
Impact Score: 74.83
Sharpness: 137.29
Contrast: 32.38
Entropy: 1.18
Edge Density: 10.65

Image: img_5.png
Impact Score: 288.33
Sharpness: 504.56
Contrast: 77.70
Entropy: 5.21
Edge Density: 52.11

Image: img_6.png
Impact Score: 228.98
Sharpness: 450.19
Contrast: 35.69
Entropy: 3.88
Edge Density: 37.29

Image: img_7.png
Impact Score: 186.71
Sharpness: 297.38
Contrast: 69.05
Entropy: 3.66
Edge Density: 24.47

Image: img_8.png
Impact Score: 240.24
Sharpness: 380.97
Contrast: 66.41
Entropy: 4.99
Edge Density: 39.35

Image: img_9.png
Impact Score: 235.52
Sharpness: 389.30
Contrast: 82.83
Entropy: 4.41
Edge Density: 30.75

============================================================
SUMMARY
============================================================
Average Impact Score: 210.85
Best Image: ./out/samples_onestep35057/img_13.png
Worst Image: ./out/samples_onestep35057/img_14.png

Saved CSV to:
./out/impact_scores.csv
working on dir (saving report there)= ./out/ddpm
============================================================
EVALUATING IMAGES
============================================================

Image: sample_0.png
Impact Score: 483.78
Sharpness: 1203.64
Contrast: 44.80
Entropy: 5.08
Edge Density: 42.83

Image: sample_1.png
Impact Score: 441.88
Sharpness: 1042.71
Contrast: 52.85
Entropy: 5.25
Edge Density: 48.31

Image: sample_2.png
Impact Score: 671.08
Sharpness: 1790.24
Contrast: 62.18
Entropy: 5.32
Edge Density: 49.00

Image: sample_3.png
Impact Score: 718.77
Sharpness: 1937.01
Contrast: 76.45
Entropy: 5.32
Edge Density: 43.77

Image: sample_4.png
Impact Score: 516.01
Sharpness: 1288.25
Contrast: 61.32
Entropy: 5.23
Edge Density: 40.53

Image: sample_5.png
Impact Score: 722.73
Sharpness: 1963.01
Contrast: 63.15
Entropy: 5.31
Edge Density: 51.36

Image: sample_6.png
Impact Score: 523.41
Sharpness: 1309.93
Contrast: 56.51
Entropy: 5.26
Edge Density: 39.84

Image: sample_7.png
Impact Score: 959.35
Sharpness: 2755.62
Contrast: 56.25
Entropy: 5.29
Edge Density: 51.24

============================================================
SUMMARY
============================================================
Average Impact Score: 629.62
Best Image: ./out/ddpm/sample_7.png
Worst Image: ./out/ddpm/sample_1.png

Saved CSV to:
./out/impact_scores.csv
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % cat >score_report.txt
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % python evualate_v2.py
working on dir (saving report there)= ./out/samples_onestep35057
============================================================
EVALUATING IMAGES
============================================================

Image: img_0.png
Impact Score: 269.94
Sharpness: 653.06
Contrast: 46.08
Entropy: 2.80
Edge Density: 30.32

Image: img_1.png
Impact Score: 175.52
Sharpness: 354.63
Contrast: 44.01
Entropy: 2.66
Edge Density: 23.59

Image: img_10.png
Impact Score: 269.94
Sharpness: 653.06
Contrast: 46.08
Entropy: 2.80
Edge Density: 30.32

Image: img_11.png
Impact Score: 175.52
Sharpness: 354.63
Contrast: 44.01
Entropy: 2.66
Edge Density: 23.59

Image: img_12.png
Impact Score: 100.55
Sharpness: 170.70
Contrast: 31.41
Entropy: 1.92
Edge Density: 16.56

Image: img_13.png
Impact Score: 331.89
Sharpness: 668.96
Contrast: 80.93
Entropy: 4.89
Edge Density: 47.00

Image: img_14.png
Impact Score: 74.83
Sharpness: 137.29
Contage: img_15.png
Impact Score: 288.33
Sharpness: 504.56
Contrast: 77.70
Entropy: 5.21
Edge Density: 52.11

Image: img_2.png
Impact Score: 100.55
Sharpness: 170.70
Contrast: 31.41
Entropy: 1.92
Edge Density: 16.56

Image: img_3.png
Impact Score: 331.89
Sharpness: 668.96
Contrast: 80.93
Entropy: 4.89
Edge Density: 47.00

Image: img_4.png
Impact Score: 74.83
Sharpness: 137.29
Contrast: 32.38
Entropy: 1.18
Edge Density: 10.65

Image: img_5.png
Impact Score: 288.33
Sharpness: 504.56
Contrast: 77.70
Entropy: 5.21
Edge Density: 52.11

Image: img_6.png
Impact Score: 228.98
Sharpness: 450.19
Contrast: 35.69
Entropy: 3.88
Edge Density: 37.29

Image: img_7.png
Impact Score: 186.71
Sharpness: 297.38
Contrast: 69.05
Entropy: 3.66
Edge Density: 24.47

Image: img_8.png
Impact Score: 240.24
Sharpness: 380.97
Contrast: 66.41
Entropy: 4.99
Edge Density: 39.35

Image: img_9.png
Impact Score: 235.52
Sharpness: 389.30
Contrast: 82.83
Entropy: 4.41
Edge Density: 30.75

=
SUMMARY
============================================================
Average Impact Score: 210.85
Best Image: ./out/samples_onestep35057/img_13.png
Worst Image: ./out/samples_onestep35057/img_14.png

Saved CSV to:
./out/impact_scores.csv
working on dir (saving report there)= ./out/ddpm
============================================================
EVALUATING IMAGES
============================================================

Image: sample_0.png
Impact Score: 483.78
Sharpness: 1203.64
Contrast: 44.80
Entropy: 5.08
Edge Density: 42.83

Image: sample_1.png
Impact Score: 441.88
Sharpness: 1042.71
Contrast: 52.85
Entropy: 5.25
Edge Density: 48.31

Image: sample_2.png
Impact Score: 671.08
Sharpness: 1790.24
Contrast: 62.18
Entropy: 5.32
Edge Density: 49.00

Image: sample_3.png
Impact Score: 718.77
Sharpness: 1937.01
Contrast: 76.45
Entropy: 5.32
Edge Density: 43.77

Image: sample_4.png
Impact Score: 516.01
Sharpness: 1288.25
Contrast: 61.32
Entropy: 5.23
Edge Density: 40.53

Image: sample_5.png
Impact Score: 722.73
Sharpness: 1963.01
Contrast: 63.15
Entropy: 5.31
Edge Density: 51.36

Image: sample_6.png
Impact Score: 523.41
Sharpness: 1309.93
Contrast: 56.51
Entropy: 5.26
Edge Density: 39.84

Image: sample_7.png
Impact Score: 959.35
Sharpness: 2755.62
Contrast: 56.25
Entropy: 5.29
Edge Density: 51.24

============================================================
SUMMARY
============================================================
Average Impact Score: 629.62
Best Image: ./out/ddpm/sample_7.png
Worst Image: ./out/ddpm/sample_1.png

Saved CSV to:
./out/impact_scores.csv

(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git add out
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git add score_report.txt 
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git status
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   ddmp_v2.py
	new file:   ddmpwork_v2.py
	new file:   evualate_v2.py
	new file:   main.py
	new file:   out/ddpm/sample_0.png
	new file:   out/ddpm/sample_1.png
	new file:   out/ddpm/sample_2.png
	new file:   out/ddpm/sample_3.png
	new file:   out/ddpm/sample_4.png
	new file:   out/ddpm/sample_5.png
	new file:   out/ddpm/sample_6.png
	new file:   out/ddpm/sample_7.png
	new file:   out/impact_scores.csv
	new file:   out/samples_onestep35057/img_0.png
	new file:   out/samples_onestep35057/img_1.png
	new file:   out/samples_onestep35057/img_10.png
	new file:   out/samples_onestep35057/img_11.png
	new file:   out/samples_onestep35057/img_12.png
	new file:   out/samples_onestep35057/img_13.png
	new file:   out/samples_onestep35057/img_14.png
	new file:   out/samples_onestep35057/img_15.png
	new file:   out/samples_onestep35057/img_2.png
	new file:   out/samples_onestep35057/img_3.png
	new file:   out/samples_onestep35057/img_4.png
	new file:   out/samples_onestep35057/img_5.png
	new file:   out/samples_onestep35057/img_6.png
	new file:   out/samples_onestep35057/img_7.png
	new file:   out/samples_onestep35057/img_8.png
	new file:   out/samples_onestep35057/img_9.png
	new file:   prep128und64.py
	new file:   rf1step_v2.py
	new file:   rf4step_v4.py
	new file:   score_report.txt
	new file:   upscaler_version2.py
	new file:   upscaler_version3.py
	new file:   upscaler_version5.py
	new file:   upscaler_version6.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.gitignore
	.python-version
	.spyproject/
	README.md
	archive/
	backup_checkpoints/
	celeb064/
	celeb128/
	checkpoints/
	data/
	fake064/
	hard_rf/
	jpg/
	logs/
	out_images/
	pyproject.toml
	real_images/
	src/
	vortrag/

(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git add vortrag
warning: adding embedded git repository: vortrag
hint: You've added another git repository inside your current repository.
hint: Clones of the outer repository will not contain the contents of
hint: the embedded repository and will not know how to obtain it.
hint: If you meant to add a submodule, use:
hint:
hint: 	git submodule add <url> vortrag
hint:
hint: If you added this path by mistake, you can remove it from the
hint: index with:
hint:
hint: 	git rm --cached vortrag
hint:
hint: See "git help submodule" for more information.
hint: Disable this message with "git config set advice.addEmbeddedRepo false"
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git add src
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git git commit -am "updated"
git: 'git' is not a git command. See 'git --help'.

The most similar command is
	init
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git commit -am "updated wo rf4step results"
[main (root-commit) 4b44b49] updated wo rf4step results
 169 files changed, 8376 insertions(+)
 create mode 100644 ddmp_v2.py
 create mode 100644 ddmpwork_v2.py
 create mode 100644 evualate_v2.py
 create mode 100644 main.py
 create mode 100644 out/ddpm/sample_0.png
 create mode 100644 out/ddpm/sample_1.png
 create mode 100644 out/ddpm/sample_2.png
 create mode 100644 out/ddpm/sample_3.png
 create mode 100644 out/ddpm/sample_4.png
 create mode 100644 out/ddpm/sample_5.png
 create mode 100644 out/ddpm/sample_6.png
 create mode 100644 out/ddpm/sample_7.png
 create mode 100644 out/impact_scores.csv
 create mode 100644 out/samples_onestep35057/img_0.png
 create mode 100644 out/samples_onestep35057/img_1.png
 create mode 100644 out/samples_onestep35057/img_10.png
 create mode 100644 out/samples_onestep35057/img_11.png
 create mode 100644 out/samples_onestep35057/img_12.png
 create mode 100644 out/samples_onestep35057/img_13.png
 create mode 100644 out/samples_onestep35057/img_14.png
 create mode 100644 out/samples_onestep35057/img_15.png
 create mode 100644 out/samples_onestep35057/img_2.png
 create mode 100644 out/samples_onestep35057/img_3.png
 create mode 100644 out/samples_onestep35057/img_4.png
 create mode 100644 out/samples_onestep35057/img_5.png
 create mode 100644 out/samples_onestep35057/img_6.png
 create mode 100644 out/samples_onestep35057/img_7.png
 create mode 100644 out/samples_onestep35057/img_8.png
 create mode 100644 out/samples_onestep35057/img_9.png
 create mode 100644 prep128und64.py
 create mode 100644 rf1step_v2.py
 create mode 100644 rf4step_v4.py
 create mode 100644 score_report.txt
 create mode 100644 src/ddpm_upscaler.py
 create mode 100644 src/imgs/060533.jpg
 create mode 100644 src/imgs/061533.jpg
 create mode 100644 src/imgs/062533.jpg
 create mode 100644 src/imgs/063533.jpg
 create mode 100644 src/imgs/064533.jpg
 create mode 100644 src/imgs/065533.jpg
 create mode 100644 src/imgs/066533.jpg
 create mode 100644 src/imgs/067533.jpg
 create mode 100644 src/imgs/068533.jpg
 create mode 100644 src/imgs/069533.jpg
 create mode 100644 src/model1.pth
 create mode 100644 src/rf_ddpmpp_celeba64_0R0.pth
 create mode 100644 src/rf_ddpmpp_celeba64_0R0_first.pth
 create mode 100644 src/rf_ddpmpp_celeba64_0R1.pth
 create mode 100644 src/rf_ddpmpp_celeba64_1R0.pth
 create mode 100644 src/rf_ddpmpp_celeba64_2R0.pth
 create mode 100644 src/rf_loader.py
 create mode 100644 src/rf_version1.py
 create mode 100644 src/samples_0ALL/img_01.png
 create mode 100644 src/samples_0ALL/img_02.png
 create mode 100644 src/samples_0ALL/img_03.png
 create mode 100644 src/samples_0ALL/img_04.png
 create mode 100644 src/samples_0ALL/img_05.png
 create mode 100644 src/samples_0ALL/img_06.png
 create mode 100644 src/samples_0ALL/img_07.png
 create mode 100644 src/samples_0ALL/img_08.png
 create mode 100644 src/samples_0ALL/img_09.png
 create mode 100644 src/samples_0ALL/img_10.png
 create mode 100644 src/samples_0ALL/img_11.png
 create mode 100644 src/samples_0ALL/img_12.png
 create mode 100644 src/samples_0ALL/img_13.png
 create mode 100644 src/samples_0ALL/img_14.png
 create mode 100644 src/samples_0ALL/img_15.png
 create mode 100644 src/samples_0ALL/img_16.png
 create mode 100644 src/samples_0K0/img_01.png
 create mode 100644 src/samples_0K0/img_02.png
 create mode 100644 src/samples_0K0/img_03.png
 create mode 100644 src/samples_0K0/img_04.png
 create mode 100644 src/samples_0K0/img_05.png
 create mode 100644 src/samples_0K0/img_06.png
 create mode 100644 src/samples_0K0/img_07.png
 create mode 100644 src/samples_0K0/img_08.png
 create mode 100644 src/samples_0K0/img_09.png
 create mode 100644 src/samples_0K0/img_10.png
 create mode 100644 src/samples_0K0/img_11.png
 create mode 100644 src/samples_0K0/img_12.png
 create mode 100644 src/samples_0K0/img_13.png
 create mode 100644 src/samples_0K0/img_14.png
 create mode 100644 src/samples_0K0/img_15.png
 create mode 100644 src/samples_0K0/img_16.png
 create mode 100644 src/samples_0K1/img_01.png
 create mode 100644 src/samples_0K1/img_02.png
 create mode 100644 src/samples_0K1/img_03.png
 create mode 100644 src/samples_0K1/img_04.png
 create mode 100644 src/samples_0K1/img_05.png
 create mode 100644 src/samples_0K1/img_06.png
 create mode 100644 src/samples_0K1/img_07.png
 create mode 100644 src/samples_0K1/img_08.png
 create mode 100644 src/samples_0K1/img_09.png
 create mode 100644 src/samples_0K1/img_10.png
 create mode 100644 src/samples_0K1/img_11.png
 create mode 100644 src/samples_0K1/img_12.png
 create mode 100644 src/samples_0K1/img_13.png
 create mode 100644 src/samples_0K1/img_14.png
 create mode 100644 src/samples_0K1/img_15.png
 create mode 100644 src/samples_0K1/img_16.png
 create mode 100644 src/samples_1K0/img_01.png
 create mode 100644 src/samples_1K0/img_02.png
 create mode 100644 src/samples_1K0/img_03.png
 create mode 100644 src/samples_1K0/img_04.png
 create mode 100644 src/samples_1K0/img_05.png
 create mode 100644 src/samples_1K0/img_06.png
 create mode 100644 src/samples_1K0/img_07.png
 create mode 100644 src/samples_1K0/img_08.png
 create mode 100644 src/samples_1K0/img_09.png
 create mode 100644 src/samples_1K0/img_10.png
 create mode 100644 src/samples_1K0/img_11.png
 create mode 100644 src/samples_1K0/img_12.png
 create mode 100644 src/samples_1K0/img_13.png
 create mode 100644 src/samples_1K0/img_14.png
 create mode 100644 src/samples_1K0/img_15.png
 create mode 100644 src/samples_1K0/img_16.png
 create mode 100644 src/samples_2K0/img_01.png
 create mode 100644 src/samples_2K0/img_02.png
 create mode 100644 src/samples_2K0/img_03.png
 create mode 100644 src/samples_2K0/img_04.png
 create mode 100644 src/samples_2K0/img_05.png
 create mode 100644 src/samples_2K0/img_06.png
 create mode 100644 src/samples_2K0/img_07.png
 create mode 100644 src/samples_2K0/img_08.png
 create mode 100644 src/samples_2K0/img_09.png
 create mode 100644 src/samples_2K0/img_10.png
 create mode 100644 src/samples_2K0/img_11.png
 create mode 100644 src/samples_2K0/img_12.png
 create mode 100644 src/samples_2K0/img_13.png
 create mode 100644 src/samples_2K0/img_14.png
 create mode 100644 src/samples_2K0/img_15.png
 create mode 100644 src/samples_2K0/img_16.png
 create mode 100644 src/samples_onestep34751/img_0.png
 create mode 100644 src/samples_onestep34751/img_1.png
 create mode 100644 src/samples_onestep34751/img_10.png
 create mode 100644 src/samples_onestep34751/img_11.png
 create mode 100644 src/samples_onestep34751/img_12.png
 create mode 100644 src/samples_onestep34751/img_13.png
 create mode 100644 src/samples_onestep34751/img_14.png
 create mode 100644 src/samples_onestep34751/img_15.png
 create mode 100644 src/samples_onestep34751/img_2.png
 create mode 100644 src/samples_onestep34751/img_3.png
 create mode 100644 src/samples_onestep34751/img_4.png
 create mode 100644 src/samples_onestep34751/img_5.png
 create mode 100644 src/samples_onestep34751/img_6.png
 create mode 100644 src/samples_onestep34751/img_7.png
 create mode 100644 src/samples_onestep34751/img_8.png
 create mode 100644 src/samples_onestep34751/img_9.png
 create mode 100644 src/samples_onestep35057/img_0.png
 create mode 100644 src/samples_onestep35057/img_1.png
 create mode 100644 src/samples_onestep35057/img_10.png
 create mode 100644 src/samples_onestep35057/img_11.png
 create mode 100644 src/samples_onestep35057/img_12.png
 create mode 100644 src/samples_onestep35057/img_13.png
 create mode 100644 src/samples_onestep35057/img_14.png
 create mode 100644 src/samples_onestep35057/img_15.png
 create mode 100644 src/samples_onestep35057/img_2.png
 create mode 100644 src/samples_onestep35057/img_3.png
 create mode 100644 src/samples_onestep35057/img_4.png
 create mode 100644 src/samples_onestep35057/img_5.png
 create mode 100644 src/samples_onestep35057/img_6.png
 create mode 100644 src/samples_onestep35057/img_7.png
 create mode 100644 src/samples_onestep35057/img_8.png
 create mode 100644 src/samples_onestep35057/img_9.png
 create mode 100644 upscaler_version2.py
 create mode 100644 upscaler_version3.py
 create mode 100644 upscaler_version5.py
 create mode 100644 upscaler_version6.py
 create mode 160000 vortrag
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git push
fatal: No configured push destination.
Either specify the URL from the command-line or configure a remote repository using

    git remote add <name> <url>

and then push using the remote name

    git push <name>

(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git push origin
fatal: The current branch main has no upstream branch.
To push the current branch and set the remote as upstream, use

    git push --set-upstream origin main

To have this happen automatically for branches without a tracking
upstream, see 'push.autoSetupRemote' in 'git help config'.

(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git push remote
fatal: The current branch main has no upstream branch.
To push the current branch and set the remote as upstream, use

    git push --set-upstream remote main

To have this happen automatically for branches without a tracking
upstream, see 'push.autoSetupRemote' in 'git help config'.

(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % git remote     
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % 
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % 
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % la    
zsh: command not found: la
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % ls
README.md		checkpoints		fake064			out			rf1step_v2.py		upscaler_version3.py
archive			data			hard_rf			out_images		rf4step_v4.py		upscaler_version5.py
backup_checkpoints	ddmp_v2.py		jpg			prep128und64.py		score_report.txt	upscaler_version6.py
celeb064		ddmpwork_v2.py		logs			pyproject.toml		src			vortrag
celeb128		evualate_v2.py		main.py			real_images		upscaler_version2.py
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % nano rf4step_v4.py 
(ddpm_rf) alex@MacBook-Pro-2 ddpm_rf % nano rf4step_v4.py

  GNU nano 9.0                                                         rf4step_v4.py                                                                    
)


# ============================================================
# MODEL
# ============================================================

model = SimpleUNet().to(DEVICE)

ema_model = copy.deepcopy(model).to(DEVICE)

for param in ema_model.parameters():
    param.requires_grad = False

ema = EMA(EMA_DECAY)

flow = RectifiedFlow(
    model=model,
    device=DEVICE
).to(DEVICE)

optimizer = torch.optim.AdamW(
    flow.parameters(),
    lr=LR
)


# ============================================================
