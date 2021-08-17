import os, shutil
from modelExceptions import ModelException
from VAEModel import VAE
from modelGenerator import ModelGenerator
import unittest

class TestModelGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ModelGenerator()

    def tearDown(self) -> None:
        self.generator = None

        folder = './testing/savedModels'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def test_load_model(self):
        # Load random file
        random_file = "RandomFileName"
        with self.assertRaises(ModelException) as exceptionOne:
            self.generator.loadModel(random_file)
        self.assertEqual(exceptionOne.exception.message, f"File {random_file} does not exist")
        
        # Load non-pytorch file
        with self.assertRaises(ModelException) as exceptionTwo:
            self.generator.loadModel("VAEModel.py")
        self.assertEqual(exceptionTwo.exception.message, "File needs to be a pytorch file")
        
        # Load Default
        self.assertEqual("defaultModels/Epochs-50.pt" ,self.generator.loadModel())

        # Load Existing model
        self.assertEqual("defaultModels/Epochs-50-Fashion.pt",self.generator.loadModel("defaultModels/Epochs-50-Fashion.pt"))

    def test_save_model(self):
        # File path not specified
        pathTo = self.generator.saveModel()
        self.assertTrue(os.path.exists(pathTo))
        # os.remove(("./"+pathTo))

        # File path specified
        self.assertEqual("testing/savedModels/testModel.pt", self.generator.saveModel("testing/savedModels/testModel.pt"))

        # Files with the same name
        pathTo = self.generator.saveModel("testing/savedModels/testModel.pt")
        self.assertTrue(os.path.exists(pathTo))

    def test_image_generation(self):
        self.generator.loadModel("defaultModels/Epochs-50-Fashion.pt")

        with self.assertRaises(ModelException) as exceptionOne:
            self.assertTrue(self.generator.generateImage([0.0],""))
        self.assertEqual(exceptionOne.exception.message, "Input vector not the same size as model's vector")

        self.assertTrue(list, type(self.generator.generateImage([0.0, 0.0, 0.0])))
        self.assertTrue(bytes, type(self.generator.generateImage([0.0, 0.0, 0.0])[0]))

        # with self.assertRaises(ModelException) as exceptionOne:
        #     self.generator.generateImage("RandomImageName.py")
        # self.assertEqual(exceptionOne.exception.message, "File extension must be either be png, jpg, jpeg")



if __name__ == "__main__":
    unittest.main()
