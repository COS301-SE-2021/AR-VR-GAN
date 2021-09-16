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
  selected: string;
  currentModelValue: any;
  modelDetails: any;
  models: any;
  fetchedData: boolean;

  constructor(
    private http: HttpClient,
    private snackBar: MatSnackBar
  ) { 
    this.selected = "";
    this.fetchedData = false;
    this.currentModel();
    this.listModels();
  }

  ngOnInit(): void {
    this.selected = "";
    this.fetchedData = false;
    this.currentModel();
    this.listModels();
  }

  currentModel(): void {
    this.http.post<any>(HOST_URL + '/model/currentModel', {
      // Empty
    }).subscribe(resp => {
      this.currentModelValue = resp['modelName'] + '.pt';
      this.selected = this.currentModelValue;
    });
  }

  listModels(): void {
    this.fetchedData = false;

    this.http.post<any>(HOST_URL + '/model/listModels', {
      'default': false,
      'saved': true
    }).subscribe(resp => {
      this.models = resp['models'];
      this.modelDetails = resp['modelDetails'];
      this.fetchedData = true;
    });
  }

  saveChanges(dataset: string): void {
    this.http.post<any>(HOST_URL + '/model/loadModel', {
      'modelName': dataset
    }).subscribe(resp => {
      if (resp['succesful'] == true) {
        this.snackBar.open(`The model was changed to ${dataset}`, "Close");
      } else {
        this.snackBar.open('The model did not change successfully', "Close");
      }

      this.currentModel();  
    });
  }
}
