import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { HOST_URL } from 'src/config/consts';

@Component({
  selector: 'app-customize',
  templateUrl: './customize.component.html',
  styleUrls: ['./customize.component.css']
})
export class CustomizeComponent implements OnInit {
  selected: string;

  constructor(
    private http: HttpClient
  ) { 
    this.selected = "option1";
  }

  ngOnInit(): void {
  }

  fetchCurrentValues(): void {
    console.log('Fetching');

    this.http.post<any>(HOST_URL + '/model/listModels', {
      'default': false,
      'saved': true
    }).subscribe(resp => {
      console.log(resp);
    });
  }

  bin2string(array: any){
    var result = "";
    for(var i = 0; i < array.length; ++i){
      result+= (String.fromCharCode(array[i]));
    }
    return result;
  }

  fetchAvailableDatasets(): void {

  }

  saveChanges(dataset: string): void {
    console.log(dataset);
  }
}
