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
    "modelName",
    "epochs",
    "latentSize",
    "beta",
    "modelType",
    "dataset",
    "fileName",
    // "current",
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

  fetchData(empty: boolean = true): void {
    this.fetchedData = false;
    
    if (empty) {
      this.dataSource = [];
    }
    
    let jsonData: modelDetails[] = [];

    this.currentModel().subscribe((currentModel) => {
      this.listModels().subscribe((listModels) => {
        for (let model in listModels['modelDetails']) {
          let newModel: modelDetails = {
            "fileName": "",
            "modelName": "",
            "epochs": 1,
            "latentSize": 3,
            "beta": "",
            "modelType": "",
            "dataset": "",
            "current": false
          };

          newModel.fileName = model;
          newModel.modelName = listModels['modelDetails'][model]['name'];
          newModel.epochs = listModels['modelDetails'][model]['epochs_trained'];
          newModel.latentSize = listModels['modelDetails'][model]['latent_vector_size'];
          newModel.beta = listModels['modelDetails'][model]['beta_value'];
          newModel.dataset = listModels['modelDetails'][model]['dataset_used'];

          if (newModel.beta == "-1") {
            newModel.modelType = "CVAE";
            newModel.beta = "N/A";
          } else {
            newModel.modelType = "Beta-CVAE";
          }

          if (newModel.modelName == currentModel['modelName']) {
            newModel.current = true;
            this.currentModelValue = newModel;
          }

          jsonData.push(newModel);
        }

        jsonData.sort(function(a,b) {
          return a.dataset < b.dataset ? -1 : 1;
        });

        this.dataSource = jsonData;
        this.fetchedData = true;
      });
    });
  }

  updateCurrentModel(modelDetails: modelDetails): void {
    this.loadModel(modelDetails.fileName).subscribe((resp) => {
      this.fetchData(false);
      this.snackBar.open(`The model was updated`,"Close");
    });
  }

  currentModel(): Observable<any> {
    return this.http.post<any>(HOST_URL + '/model/currentModel', {
      // Empty
    });
  }

  listModels(): Observable<any> {
    return this.http.post<any>(HOST_URL + '/model/listModels', {
      'default': true,
      'saved': true
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
  beta: string;
  modelType: string;
  dataset: string;
  current: boolean;
}