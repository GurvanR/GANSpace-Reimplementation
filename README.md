# GANSpace-Reimplementation


# ImageCreator.ipynb 
- Allows to create sets of images **with their correspoding latent codes in Z** and save them on your Drive.
- It is recommended to let them on your Drive and working on your Drive for next because of time duration of downloading and uploading on Colab.
- It is recommended to do batch of less than 5k images (N <= 5_000) because it will take around 15 minutes. It is good to have at least 10k.
- Be careful to do it on a GPU to save some time.
  
# ImageDiscriminator.ipynb
- Allows to discriminate each image according to an attribute.
- The list of the 40 attributes is available on the notebook.
- First you have to pass every images you have made from the ImageCreator nb to the discriminator, it will give a score for each attibute for each image.
- **For each** attribute you choose, it creates a labeled dataset as follows :
  - The top 2% of the images that have the highest score for that attribute, labeled as **1**.
  - The top 2% of the image that have the lowest score, labeled as **-1**.
  - Such that each dataset you create has two classes and a size of 4% of the total size of your original set of images.
  - The percentage by default is 2% but you can choose it in the notebook.

# InterFaceGAN.ipynb
