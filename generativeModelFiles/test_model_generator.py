from modelExceptions import ModelException
from VAEModel import VAE
from modelGenerator import ModelGenerator
import unittest

class TestModelGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ModelGenerator()

    def tearDown(self) -> None:
        self.generator = None

    def test_load_model(self):
        self.assertTrue(self.generator.loadModel())
        with self.assertRaises(ModelException) as exceptionOne:
            self.generator.loadModel("RandomFileName")
        self.assertEqual(exceptionOne.exception.message, "File RandomFileName does not exist")
        
        with self.assertRaises(ModelException) as exceptionTwo:
            self.generator.loadModel("VAEModel.py")
        self.assertEqual(exceptionTwo.exception.message, "File needs to be a pytorch file")
        
        self.assertEqual("defaultModels/Epochs-50.pt" ,self.generator.loadModel())
        self.assertEqual("defaultModels/Epochs-50.pt",self.generator.loadModel("defaultModels/Epochs-100.pt"))

    def test_save_model(self):
        self.assertTrue(self.generator.saveModel())
        self.assertTrue(self.generator.saveModel("testing/savedModels/testModel.pt"))
        self.assertTrue(self.generator.saveModel("testing/savedModels/testModel.pt"))

    def test_image_generation(self):
        self.generator.loadModel("defaultModels/Epochs-100.pt")
        self.assertTrue(self.generator.generateImage(""))

        self.assertTrue(self.generator.generateImage("testing/images/testingImages.png"))
        self.assertTrue(self.generator.generateImage("testing/images/testingImages.jpeg"))
        self.assertTrue(self.generator.generateImage("testing/images/testingImages.jpg"))

        # with self.assertRaises(ModelException) as exceptionOne:
        #     self.generator.generateImage("RandomImageName.py")
        # self.assertEqual(exceptionOne.exception.message, "File extension must be either be png, jpg, jpeg")



if __name__ == "__main__":
    unittest.main()

