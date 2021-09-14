import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-customize',
  templateUrl: './customize.component.html',
  styleUrls: ['./customize.component.css']
})
export class CustomizeComponent implements OnInit {
  selected: string;

  constructor() { 
    this.selected = "option1";
  }

  ngOnInit(): void {
  }

  fetchCurrentValues(): void {

  }

  fetchAvailableDatasets(): void {

  }

  saveChanges(dataset: string): void {
    console.log(dataset);
  }
}
