import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { FormControl, Validators } from '@angular/forms';
import { MatSnackBar } from '@angular/material/snack-bar';
import { CookieService } from 'ngx-cookie-service';
import { HOST_URL } from 'src/config/consts';
import { CustomizeComponent } from '../customize/customize.component';

@Component({
  selector: 'app-train-model',
  templateUrl: './train-model.component.html',
  styleUrls: ['./train-model.component.css'],
  providers: [CustomizeComponent]
})
export class TrainModelComponent implements OnInit {
  modelName: string;
  trainingEpochs: number;
  readonly latentSize: number = 3;
  datasetName: string;
  beta: number;
  modelType: string;

  modelNameFormControl = new FormControl('', [
    Validators.required
  ]);
  
  constructor(
    private http: HttpClient, 
    private snackBar: MatSnackBar,
    private cookieService: CookieService,
    private customizeComponent: CustomizeComponent
  ) { 
    this.modelName = "Default";
    this.trainingEpochs = 5;
    this.datasetName = "mnist";
    this.beta = 1;
    this.modelType = "CAE";
  }

  ngOnInit(): void {
    this.modelName = "Default";
    this.trainingEpochs = 5;
    this.datasetName = "mnist";
    this.beta = 1;
    this.modelType = "CAE";
  }

  trainModel(): void {
    let modelName = this.modelName;
    let trainingEpochs = this.trainingEpochs;
    let beta = this.beta;
    let datasetName = this.datasetName;
    let modelType = this.modelType;

    if (modelName.length < 1) {
      this.snackBar.open("Please enter a valid model name", "Close");
      return;
    }

    if ((trainingEpochs < 1) || (trainingEpochs > 20)) {
      this.snackBar.open("Please use an epoch value in the range [1,20]", "Close");
      return;
    }

    if ((beta < 1) || (beta > 10)) {
      this.snackBar.open("Please use an beta value in the range [1,10]", "Close");
      return;
    }

    if (!((datasetName == "mnist") || (datasetName == "fashion") || (datasetName == "cifar10"))) {
      this.snackBar.open("Please choose a valid dataset", "Close");
      return;
    }

    if (!((modelType == "CAE") || (modelType == "CVAE"))) {
      this.snackBar.open("Please choose a valid model type", "Close");
      return;
    }

    this.customizeComponent.listModels().subscribe((modelList) => {
      for (let value in modelList['modelDetails']) {
        if (modelList['modelDetails'][value]['name'] === modelName) {
          this.snackBar.open("The model name you chose is already taken", "Close");
          return;
        }
      }

      this.snackBar.open("You will be emailed when your model is finished training", "Close");

      let options = {
        'modelName': modelName,
        'trainingEpochs': trainingEpochs,
        'latentSize': this.latentSize,
        'datasetName': datasetName,
        'beta': beta,
        'modelType': modelType,
        'jwtToken': this.cookieService.get('jwtToken')
      }

      this.http.post<any>(HOST_URL + '/model/trainModel/', options).subscribe((resp) => {
        this.snackBar.open(`The model ${modelName} is ready.`, "Close");
      });
    });
  }
}