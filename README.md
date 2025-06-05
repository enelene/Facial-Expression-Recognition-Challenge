# Facial Expression Recognition Challenge - იტერაციული ექსპერიმენტები

მოცემული რეპოზიტორია ასახავს Convolutional Neural Network (CNN) მოდელების შექმნისა და შეფასების პროცესს Kaggle-ის competition-ისთვის "Challenges in Representation Learning: Facial Expression Recognition Challenge". 

**სამუშაო გარემო:** Google Colab (GPU-თი)
**Data:** FER2013 (`icml_face_data.csv`-დან, დაყოფილია Training, PublicTest და PrivateTest ნაწილებად)
**ძირითადი ბიბლიოთეკები:** PyTorch, Weights & Biases, Pandas, NumPy
**wandb link** : (https://wandb.ai/egabe21-free-university-of-tbilisi-/facial-expression-recognition-challenge)
https://api.wandb.ai/links/egabe21-free-university-of-tbilisi-/7deagbol
-----

## ექსპერიმენტები 

### იტერაცია 1: საბაზისო მოდელი - `SimpleCNN_Baseline`

  * **მიზანი:** საწყისი, მარტივი CNN არქიტექტურის აწყობა.
  * **არქიტექტურა (`SimpleCNN`):**
      * Input: 1x48x48 ნაცრისფერი სურათები.
      * Conv Block 1: `Conv2d(1, 16, ks=3, pad=1) -> ReLU -> MaxPool2d(2,2)` (Output: 16x24x24)
      * Conv Block 2: `Conv2d(16, 32, ks=3, pad=1) -> ReLU -> MaxPool2d(2,2)` (Output: 32x12x12)
      * Flatten: Output size 32\*12\*12 = 4608
      * FC Block: `Linear(4608, 128) -> ReLU -> Dropout(p=0.5) -> Linear(128, 7)`
  * **Hyperparameters:**
      * Learning Rate: 0.001
      * Optimizer: Adam
      * Loss Function: CrossEntropyLoss
      * Epochs: 15
      * Batch Size: 64
      * Dropout (FC): 0.5
  * **Wandb Run:** `[run-SimpleCNN_Baseline-20250602-185957](https://wandb.ai/egabe21-free-university-of-tbilisi-/facial-expression-recognition-challenge/runs/jnvyn0a6)` 
  * **დაკვირვებები (ლურჯი ხაზი გრაფიკებზე):**
      * Training Accuracy: დაიწყო დაახლოებით 0.33-ით, დასრულდა 0.53-თან ახლოს.
      * Validation Accuracy: დაიწყო დაახლოებით 0.42-ით, პიკს მიაღწია 0.54-0.55 ფარგლებში.
      * Training Loss: დაიწყო \~1.69-ით, დასრულდა \~1.23-ით.
      * Validation Loss: დაიწყო \~1.52-ით, დასრულდა \~1.21-ით.
      * მოდელი აშკარად სწავლობდა, რაც გამოიხატა training და validation loss-ების შემცირებით და accuracy-ების ზრდით.
      * training და validation accuracy-ს შორის სხვაობა მცირე იყო, რაც მიუთითებდა, რომ ამ ეტაპზე მნიშვნელოვანი overfitting არ შეინიშნებოდა.
  * **ანალიზი:**
      * მოდელი შესაძლოა **underfitting**-ს განიცდიდა მისი შეზღუდული შესაძლებლობების გამო. სავარაუდოდ, ის არ იყო საკმარისად კომპლექსური სახის გამომეტყველების რთული ნიშნების ამოსაცნობად.
  * **გადაწყვეტილება შემდეგი ნაბიჯისთვის:** მოდელის სირთულის გაზრდა, რათა პოტენციურად გაუმჯობესდეს მისი მონაცემებიდან სწავლის უნარი.

-----

### იტერაცია 2: სირთულის გაზრდა - `DeeperCNN_v1` (15 Epochs)

  * **მიზანი:** შეფასებულიყო, შეძლებდა თუ არა უფრო კომპლექსური CNN არქიტექტურა საბაზისო მოდელზე უკეთესი შედეგის მიღწევას.
  * **ცვლილებები საბაზისო მოდელთან შედარებით:**
      * გაიზარდა ფილტრების რაოდენობა conv ფენებში: Conv1 (1-\>32), Conv2 (32-\>64).
      * დაემატა მესამე convolutional block: `Conv2d(64, 128, ks=3, pad=1) -> ReLU -> MaxPool2d(2,2)` (Output: 128x6x6).
      * გაიზარდა FC1 ერთეულების რაოდენობა: `Linear(128*6*6, 256)`.
      * Dropout (p=0.5) და სხვა hyperparameters უცვლელი დარჩა საწყისი შედარებისთვის.
  * **არქიტექტურა (`DeeperCNN`):**
      * Conv Block 1: `Conv2d(1, 32, ks=3, pad=1) -> ReLU -> MaxPool2d(2,2)` (32x24x24)
      * Conv Block 2: `Conv2d(32, 64, ks=3, pad=1) -> ReLU -> MaxPool2d(2,2)` (64x12x12)
      * Conv Block 3: `Conv2d(64, 128, ks=3, pad=1) -> ReLU -> MaxPool2d(2,2)` (128x6x6)
      * Flatten: Output size 128\*6\*6 = 4608
      * FC Block: `Linear(4608, 256) -> ReLU -> Dropout(p=0.5) -> Linear(256, 7)`
  * **Hyperparameters:** იგივე, რაც იტერაცია 1-ში, არქიტექტურის სპეციფიკის გარდა. Epochs: 15.
  * **Wandb Run:** `[run-DeeperCNN_v1-20250605-141737](https://wandb.ai/egabe21-free-university-of-tbilisi-/facial-expression-recognition-challenge/runs/cht7k3iq)` 
  * **დაკვირვებები (წითელი ხაზი გრაფიკებზე):**
      * Training Accuracy: დაიწყო \~0.32-ით, დასრულდა \~0.61-ით.
      * Validation Accuracy: დაიწყო \~0.42-ით, პიკს მიაღწია \~0.58-0.59. ეს შესამჩნევი გაუმჯობესება იყო საწყის მოდელთან შედარებით.
      * Training Loss: დაიწყო \~1.68-ით, დასრულდა \~1.04-ით.
      * Validation Loss: დაიწყო \~1.51-ით, დასრულდა \~1.10-ით.
      * ორივე, training და validation, მეტრიკა უკეთესი იყო, ვიდრე `SimpleCNN`-ის შემთხვევაში.
      * training accuracy-სა (\~0.61) და validation accuracy-ს (\~0.58) შორის სხვაობა კვლავ შედარებით მცირე იყო, რაც მძიმე overfitting-ის არარსებობაზე მიუთითებდა.
  * **ანალიზი:**
      * მოდელის არქიტექტურის გართულებამ მნიშვნელოვნად გააუმჯობესა წარმადობა, რაც ადასტურებდა, რომ საბაზისო მოდელს მართლაც შეზღუდული შესაძლებლობები ჰქონდა. `DeeperCNN_v1` უკეთ ახერხებდა რელევანტური ნიშნების სწავლას.
  * **გადაწყვეტილება შემდეგი ნაბიჯისთვის:**
    1.  გამოკვლეულიყო, შეძლებდა თუ არა ეს გაუმჯობესებული არქიტექტურა (`DeeperCNN_v1`) კიდევ უკეთესი შედეგების მიღწევას მეტი სატრენინგო epoch-ით.
    2.  გამოკვლეულიყო ბევრად დიდი მოდელის ქცევა overfitting-ისა და regularization-ის ეფექტის გასაგებად.

-----

### იტერაცია 3: Overfitting-ის კვლევა - `OverfitCNN_v1_NoDropout`

  * **მიზანი:** მიზანმიმართულად შექმნილიყო და გაწვრთნილიყო ძალიან კომპლექსური მოდელი მინიმალური regularization-ით, რათა დაკვირვებოდით და გაგვეანალიზებინა overfitting-ის ქცევა.
  * **ცვლილებები `DeeperCNN_v1`-თან შედარებით:**
      * მნიშვნელოვნად გაიზარდა ფილტრების რაოდენობა: Conv1 (1-\>64), Conv2 (64-\>128), Conv3 (128-\>256).
      * დაემატა მეოთხე convolutional block: `Conv2d(256, 512, ks=3, pad=1) -> ReLU -> MaxPool2d(2,2)` (Output: 512x3x3).
      * მნიშვნელოვნად გაიზარდა FC1 ერთეულების რაოდენობა: `Linear(512*3*3, 1024)`.
      * **სრულად ამოღებულ იქნა FC Dropout** ამ ექსპერიმენტისთვის, overfitting-ის წასახალისებლად.
  * **არქიტექტურა (`OverfitCNN`):**
      * Conv1: (1-\>64) -\> ReLU -\> Pool (64x24x24)
      * Conv2: (64-\>128) -\> ReLU -\> Pool (128x12x12)
      * Conv3: (128-\>256) -\> ReLU -\> Pool (256x6x6)
      * Conv4: (256-\>512) -\> ReLU -\> Pool (512x3x3)
      * Flatten: Output size 512\*3\*3 = 4608
      * FC Block: `Linear(4608, 1024) -> ReLU -> Linear(1024, 7)` (No Dropout)
  * **Hyperparameters:** LR=0.001, Adam, CrossEntropyLoss, Batch Size=64. **Epochs: 25**. Dropout=0.0.
  * **Wandb Run:** `[run-OverfitCNN_v1_NoDropout-20250605-143435](https://wandb.ai/YOUR_WANDB_USERNAME/YOUR_WANDB_PROJECT_NAME/runs/tbq6yyct)` 
  * **დაკვირვებები (მწვანე ხაზი გრაფიკებზე):**
      * Training Accuracy: ძალიან სწრაფად გაიზარდა და 25-ე epoch-ისთვის მიაღწია \~0.80-ს.
      * Validation Accuracy: პიკს მიაღწია დაახლოებით 0.60-0.61 ფარგლებში (დაახლ. 13-16 epoch-ები) და შემდეგ დაიწყო კლება.
      * Training Loss: მკვეთრად შემცირდა დაბალ მნიშვნელობამდე (დაახლოებით 0.54).
      * Validation Loss: თავდაპირველად შემცირდა დაახლოებით 1.11-მდე (მე-7 epoch-ა), მაგრამ შემდეგ **დაიწყო სტაბილური ზრდა** და 25-ე epoch-ისთვის მიაღწია \~1.39-ს.
      * გაჩნდა მნიშვნელოვანი სხვაობა training accuracy-სა და validation accuracy-ს შორის.
  * **ანალიზი:**
      * მოდელმა აჩვენა **overfitting-ის კლასიკური ნიშნები**. validation loss-ის ზრდა, მაშინ როცა training loss კვლავ მცირდება, ამის მთავარი ინდიკატორია.
      * validation accuracy (დაახლოებით 0.60-0.61) ოდნავ უკეთესი იყო, ვიდრე `DeeperCNN_v1`-ის 15 epoch-ზე, რაც მიუთითებდა, რომ გაზრდილ ეპოქებს *შეუძლია* უკეთესი გადაწყვეტილებების პოვნა, მაგრამ regularization-ის გარეშე უაზრობაა.
  * **გადაწყვეტილება შემდეგი ნაბიჯისთვის:** `OverfitCNN` არქიტექტურაზე regularization-ის ტექნიკების (Batch Normalization, Dropout) გამოყენება, რათა შევეცადოთ overfitting-ის შემცირება და განზოგადების გაუმჯობესება.

-----

### იტერაცია 4: Overfitting-ის შემცირება - `RegularizedOverfitCNN_v1`

  * **მიზანი:** Regularization-ის ტექნიკების გამოყენება მე-3 იტერაციის overfitting-ზე მიდრეკილი `OverfitCNN` არქიტექტურისთვის და განზოგადების გაუმჯობესებაზე დაკვირვება.
  * **ცვლილებები `OverfitCNN_v1_NoDropout`-თან შედარებით:**
      * გამოყენებულ იქნა იგივე საბაზისო convolutional და FC სტრუქტურა.
      * **დაემატა `BatchNorm2d`** ყოველი convolutional layer-ის შემდეგ (ReLU-მდე).
      * **დაემატა `BatchNorm1d`** პირველი fully connected layer-ის შემდეგ (ReLU-მდე).
      * **ხელახლა დაინერგა `Dropout(p=0.5)`** პირველი fully connected layer-ის activation-ის შემდეგ.
  * **არქიტექტურა (`RegularizedOverfitCNN`):**
      * Conv1: (1-\>64) -\> BN -\> ReLU -\> Pool
      * Conv2: (64-\>128) -\> BN -\> ReLU -\> Pool
      * Conv3: (128-\>256) -\> BN -\> ReLU -\> Pool
      * Conv4: (256-\>512) -\> BN -\> ReLU -\> Pool
      * Flatten
      * FC Block: `Linear(4608, 1024) -> BN -> ReLU -> Dropout(p=0.5) -> Linear(1024, 7)`
  * **Hyperparameters:** LR=0.001, Adam, CrossEntropyLoss, Batch Size=64. **Epochs: 25**. Dropout=0.5. BatchNorm=True.
  * **Wandb Run:** `[run-RegularizedOverfitCNN_v1-20250605-145737](https://wandb.ai/YOUR_WANDB_USERNAME/YOUR_WANDB_PROJECT_NAME/runs/4db36ht3)`
  * **დაკვირვებები (იასამნისფერი ხაზი გრაფიკებზე):**
      * Training Accuracy: სტაბილურად გაიზარდა და 25-ე epoch-ისთვის მიაღწია ძალიან მაღალ \~0.86-ს.
      * Validation Accuracy: აჩვენა მნიშვნელოვანი გაუმჯობესება არარეგულარიზებულ ვერსიასთან შედარებით, პიკს მიაღწია დაახლოებით **0.65**-ს (დაახლ. 17-21 epoch-ები). ეს არის საუკეთესო validation accuracy აქამდე.
      * Training Loss: მნიშვნელოვნად შემცირდა დაახლოებით 0.37-მდე.
      * Validation Loss: შემცირდა მინიმუმამდე, დაახლოებით 1.04-1.06 (მე-7 epoch-ა) და შემდეგ დარჩა შედარებით სტაბილური ან ოდნავ გაიზარდა \~1.1-1.2-მდე, მაგრამ არ აჩვენა ისეთი დრამატული ზრდა, როგორიც არარეგულარიზებულ `OverfitCNN`-ში.
      * training accuracy-სა და validation accuracy-ს შორის სხვაობა კვლავ არსებობდა (0.86 vs 0.65), რაც მიუთითებდა, რომ მოდელი შესაძლოა კვლავ გარკვეულწილად overfitting-ს განიცდიდა მისი დიდი შესაძლებლობების გამო, მაგრამ regularization-მა მნიშვნელოვნად გააუმჯობესა მისი განზოგადების უნარი.
  * **ანალიზი:**
      * Batch Normalization და Dropout  **ძალიან ეფექტური** აღმოჩნდა მე-3 იტერაციის მძიმე overfitting-ის შესამცირებლად.
      * მოდელმა მიაღწია საუკეთესო validation accuracy-ს აქამდე (\~0.65), რაც აჩვენებს, რომ მაღალი შესაძლებლობების მოდელს *შეუძლია* კარგად იმუშაოს სათანადო regularization-ის პირობებში.
      * დარჩენილი overfitting მიუთითებს, რომ regularization-ის სიძლიერის შემდგომი დახვეწა (მაგ., dropout rate, weight decay value) ან სხვა ტექნიკების გამოკვლევა შეიძლება სასარგებლო იყოს, მაგრამ აქ გავჩერდი.
  * **გადაწყვეტილება შემდეგი ნაბიჯისთვის:** მიუხედავად იმისა, რომ ეს რეგულარიზებული კომპლექსური მოდელი კარგია, დავუბრუნდეთ `DeeperCNN_v1`-ს (რომელმაც ადრე კარგი განზოგადება აჩვენა) და ვნახოთ, შეძლებს თუ არა ის უბრალოდ მეტი epoch-ით წვრთნით შედარებადი ან უკეთესი შედეგების მიღწევას `OverfitCNN` არქიტექტურის შესაძლებლობების გარეშე.

### იტერაცია 5: `DeeperCNN_v1`-ის მეტი Epoch-ით წვრთნა

  * **მიზანი:** დადგინდეს, შეძლებს თუ არა `DeeperCNN_v1` არქიტექტურა (მე-2 იტერაციიდან) უკეთესი წარმადობის მიღწევას უბრალოდ უფრო ხანგრძლივი დროის განმავლობაში წვრთნით.
  * **არქიტექტურა (`DeeperCNN`):** იგივე, რაც მე-2 იტერაციაში.
  * **Hyperparameters:** იგივე, რაც მე-2 იტერაციაში, გარდა **Epochs: 30**.
  * **Wandb Run:** `[run-DeeperCNN_v1_MoreEpochs-20250605-150511](https://wandb.ai/YOUR_WANDB_USERNAME/YOUR_WANDB_PROJECT_NAME/runs/9yqwhg11)` 
  * **დაკვირვებები (ვარდისფერი/იასამნისფერი ხაზი გრაფიკებზე):**
      * Training Accuracy: განაგრძო გაუმჯობესება 15 epoch-ის შემდეგ და 30-ე epoch-ისთვის მიაღწია \~0.62-ს.
      * Validation Accuracy: გაიზარდა \~0.58-დან (ამ არქიტექტურისთვის 15 epoch-ზე) და პიკს მიაღწია დაახლოებით **0.60**-ს 30-ე epoch-ისთვის.
      * Training Loss: განაგრძო კლება და დასრულდა დაახლოებით 0.99-ით.
      * Validation Loss: განაგრძო კლება და დასრულდა დაახლოებით 1.10-ით.
      * მოდელმა განაგრძო სწავლა და გააუმჯობესა თავისი განზოგადება validation set-ზე მეტი epoch-ით.
      * training accuracy-სა და validation accuracy-ს შორის სხვაობა მცირე დარჩა, რაც მიუთითებს, რომ გახანგრძლივებული წვრთნით ამ არქიტექტურისთვის მნიშვნელოვანი overfitting არ მომხდარა.
  * **ანალიზი:**
      * `DeeperCNN_v1`-ის მეტი epoch-ით (30 vs 15) წვრთნამ მცირედი გაუმჯობესება მოგვცა validation accuracy-ში (\~0.58-დან \~0.60-მდე).
      * ეს არქიტექტურა საკმაოდ მდგრადი ჩანს overfitting-ის მიმართ გახანგრძლივებული წვრთნის პირობებშიც კი, სავარაუდოდ მისი უფრო ზომიერი სირთულისა და dropout-ის არსებობის გამო `OverfitCNN`-თან შედარებით.
      * მიუხედავად იმისა, რომ მან ვერ მიაღწია `RegularizedOverfitCNN_v1`-ის \~0.65-ს, მან მიაღწია თავის \~0.60 შედეგს უფრო მარტივი არქიტექტურით.
  * **გადაწყვეტილება შემდეგი ნაბიჯისთვის (შემდგომი კვლევის იდეები):**
      * `DeeperCNN_v1`-ზე Batch Normalization-ის დამატება და 30 epoch-ით წვრთნა, რათა ვნახოთ, შეძლებს თუ არა ის `RegularizedOverfitCNN_v1`-სთან მიახლოებას.
      * learning rate scheduler-ების ექსპერიმენტირება უფრო ხანგრძლივი წვრთნისას.
      * regularization-ის პარამეტრების (dropout, weight decay) დახვეწა `RegularizedOverfitCNN_v1` არქიტექტურაზე. (next time)
