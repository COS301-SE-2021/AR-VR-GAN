import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-coords',
  templateUrl: './coords.component.html',
  styleUrls: ['./coords.component.css']
})
export class CoordsComponent implements OnInit {
  public image: any;
  private xMove: number;
  private yMove: number;

  constructor(private http: HttpClient) {
    this.xMove = 0;
    this.yMove = 0;
  }

  ngOnInit(): void {}

  postCoords(x: string, y: string, z: string) {
    var xN: number = +x;
    var yN: number = +y;
    var zN: number = +z;
    
    var data = [
      xN, yN, zN
    ];

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

  trackCoords(event: MouseEvent) {
    var x = (event.offsetX - 150) / 150;
    var y = (event.offsetY - 150) / 150;

    this.xMove += Math.abs(event.movementX);
    this.yMove += Math.abs(event.movementY);

    if (this.xMove + this.yMove > 20) {
      this.postCoords(x.toString(), y.toString(), '0');
      this.xMove = 0;
      this.yMove = 0;
    }
  }
}