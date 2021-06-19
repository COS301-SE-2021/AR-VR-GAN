import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-coords',
  templateUrl: './coords.component.html',
  styleUrls: ['./coords.component.css']
})
export class CoordsComponent implements OnInit {
  constructor(private http: HttpClient) { }

  ngOnInit(): void {
  }

  postCoords() {
    let x = this.getElementByID("x");
    let y = this.getElementByID("y");
    let z = this.getElementByID("z");

    var coords = 
      {
        "data": {
          x,
          y,
          z
        }
      };

    this.http.post<any>('http://localhost:3000/model/testGRPC', { coords }).subscribe(data => {
      console.log(data.sum);
    })
  }

  getElementByID(id: string){
    return document.getElementById(id);
  }





}