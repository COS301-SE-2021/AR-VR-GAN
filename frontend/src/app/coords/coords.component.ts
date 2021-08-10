import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-coords',
  templateUrl: './coords.component.html',
  styleUrls: ['./coords.component.css']
})
export class CoordsComponent implements OnInit {
  public response: any;
  public image: any;
  constructor(private http: HttpClient) { 
  }

  ngOnInit(): void {
  }

  postCoords(x: string, y: string, z: string) {
    var xN: number = +x;
    var yN: number = +y;
    var zN: number = +z;
    var data = [
      xN, yN, zN
    ];

    
    //fetching sum of coordinates
    this.http.post<any>('http://localhost:3000/model/testGRPC', { data: data}).subscribe(resp => {
      this.response = resp.sum;
    })


    //fetching an image from the server based on coordinates
    //const headers = new HttpHeaders().set('Content-Type', 'application/json');
    this.http.post('http://localhost:3000/upload/getImageFromCoordinates', { data: data }, { responseType: 'blob'}).subscribe(resp => {
      this.createImageFromBlob(resp);
    })
  }

  postNumber(x: number, y: number) {
    var xN: number = +x;
    var yN: number = +y;
    var zN: number = +0;
    var data = [
      xN, yN, zN
    ];

    
    //fetching sum of coordinates
    this.http.post<any>('http://localhost:3000/model/testGRPC', { data: data}).subscribe(resp => {
      this.response = resp.sum;
    })


    //fetching an image from the server based on coordinates
    //const headers = new HttpHeaders().set('Content-Type', 'application/json');
    this.http.post('http://localhost:3000/upload/getImageFromCoordinates', { data: data }, { responseType: 'blob'}).subscribe(resp => {
      this.createImageFromBlob(resp);
    })
  }

  createImageFromBlob(blob: Blob) {
    let reader = new FileReader();
    reader.addEventListener("load", () => {
        this.image = reader.result;
   }, false);

   if (blob) {
      reader.readAsDataURL(blob);
   }
  }

  trackCoords(event: MouseEvent){
    console.log(event.clientX + " " + event.clientY);

    this.postNumber(event.clientX, event.clientY);

  }

}