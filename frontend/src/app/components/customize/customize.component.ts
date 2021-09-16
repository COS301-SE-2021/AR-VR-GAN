import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { HOST_URL } from 'src/config/consts';

@Component({
  selector: 'app-customize',
  templateUrl: './customize.component.html',
  styleUrls: ['./customize.component.css']
})
export class CustomizeComponent implements OnInit {
  currentModelValue: any;
  listModelsValue: any;
  fetchedData: boolean;
  selected: string;

  constructor(
    private http: HttpClient,
    private snackBar: MatSnackBar
  ) { 
    this.fetchedData = false;
    this.selected = "";
    this.listModelsValue = {
      "models": []
    };

    this.listModels();
    this.currentModel();
  }

  ngOnInit(): void {
    this.fetchedData = false;
    this.selected = "";
    this.listModelsValue = {
      "models": []
    };
    
    this.listModels();
    this.currentModel();
  }

  currentModel(): void {
    this.http.post<any>(HOST_URL + '/model/currentModel', {
      // Empty
    }).subscribe(resp => {
      this.currentModelValue = resp;
    });
  }

  listModels(): void {
    this.fetchedData = false;

    this.http.post<any>(HOST_URL + '/model/listModels', {
      'default': false,
      'saved': true
    }).subscribe(resp => {
      this.listModelsValue = resp;
      this.fetchedData = true;
    });
  }

  saveChanges(dataset: string): void {
    this.http.post<any>(HOST_URL + '/model/loadModel', {
      'modelName': dataset
    }).subscribe(resp => {
      if (resp['succesful'] == true) {
        this.snackBar.open(`The model was changed to ${this.listModelsValue['modelDetails'][dataset]['name']}`, "Close");
      } else {
        this.snackBar.open('The model did not change successfully', "Close");
      }

      this.currentModel();  
    });
  }
}
