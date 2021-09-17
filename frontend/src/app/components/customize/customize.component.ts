import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Observable } from 'rxjs';
import { HOST_URL } from 'src/config/consts';

@Component({
  selector: 'app-customize',
  templateUrl: './customize.component.html',
  styleUrls: ['./customize.component.css']
})
export class CustomizeComponent implements OnInit {
  fetchedData: boolean = false;
  dataSource: modelDetails[] = [];
  currentModelValue: modelDetails | any;

  readonly displayedColumns: string[] = [
    "fileName",
    "modelName",
    "epochs",
    "latentSize",
    "beta",
    "dataset",
    "current"
  ];

  constructor(
    private http: HttpClient,
    private snackBar: MatSnackBar
  ) { 
    this.fetchData();
  }

  ngOnInit(): void {
    this.fetchData();
  }

  fetchData(): void {
    this.fetchedData = false;
    let jsonData: modelDetails[] = [];

    this.currentModel().subscribe((currentModel) => {
      this.listModels(false, true).subscribe((listModels) => {
        for (let model in listModels['modelDetails']) {
          let newModel: modelDetails = {
            "fileName": "",
            "modelName": "",
            "epochs": 1,
            "latentSize": 3,
            "beta": 1,
            "dataset": "",
            "current": false
          };

          newModel.fileName = model;
          newModel.modelName = listModels['modelDetails'][model]['name'];
          newModel.epochs = listModels['modelDetails'][model]['epochs_trained'];
          newModel.latentSize = listModels['modelDetails'][model]['latent_vector_size'];
          newModel.beta = listModels['modelDetails'][model]['beta_value'];
          newModel.dataset = listModels['modelDetails'][model]['dataset_used'];

          if (newModel.modelName == currentModel['modelName']) {
            newModel.current = true;
            this.currentModelValue = newModel;
          }

          jsonData.push(newModel);
        }

        this.dataSource = jsonData;
        this.fetchedData = true;
      });
    });
  }

  updateCurrentModel(modelDetails: modelDetails): void {
    this.loadModel(modelDetails.fileName).subscribe((resp) => {
      this.fetchData();
      this.snackBar.open(`The model was set to ${modelDetails.modelName}`,"Close");
    });
  }

  currentModel(): Observable<any> {
    return this.http.post<any>(HOST_URL + '/model/currentModel', {
      // Empty
    });
  }

  listModels(defaultValue: boolean, savedValue: boolean): Observable<any> {
    return this.http.post<any>(HOST_URL + '/model/listModels', {
      'default': defaultValue,
      'saved': savedValue
    });
  }

  loadModel(fileName: string): Observable<any> {
    return this.http.post<any>(HOST_URL + '/model/loadModel', {
      'modelName': fileName
    });
  }
}

export interface modelDetails {
  fileName: string;
  modelName: string;
  epochs: number;
  latentSize: number;
  beta: number;
  dataset: string;
  current: boolean;
}