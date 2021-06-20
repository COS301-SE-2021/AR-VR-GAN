import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-coords',
  templateUrl: './coords.component.html',
  styleUrls: ['./coords.component.css']
})
export class CoordsComponent implements OnInit {
  public response: any;
  constructor(private http: HttpClient) { 
  }

  ngOnInit(): void {
  }

  postCoords(x: string, y: string, z: string) {
    console.log(x + " " + y + " " + z);
    var xN: number = +x;
    var yN: number = +y;
    var zN: number = +z;
    var data = [
      xN, yN, zN
    ];

    this.http.post<any>('http://localhost:3000/model/testGRPC', { data: data}).subscribe(resp => {
      console.log(resp.sum);
      this.response = resp.sum;
    })
  }

  getElementByID(id: string){
    return document.getElementById(id);
  }





}