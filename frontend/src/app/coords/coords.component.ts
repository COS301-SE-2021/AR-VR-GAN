import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-coords',
  templateUrl: './coords.component.html',
  styleUrls: ['./coords.component.css']
})
export class CoordsComponent implements OnInit {
  private xMove: number;
  private yMove: number;
  private busy: boolean;
  public image: any;
  public x: number;
  public y: number;
  public z: number;

  constructor(private http: HttpClient) {
    this.x = 0;
    this.y = 0;
    this.z = 0;

    this.xMove = 0;
    this.yMove = 0;
    this.busy = false;

    this.postCoords(this.x, this.y, this.z);
  }

  ngOnInit(): void {}

  postCoords(x: number, y: number, z: number) {
    var data = [
      x, y, z
    ];

    this.busy = true;

    this.http.post('http://localhost:3000/upload/getImageFromCoordinates', { data: data }, { responseType: 'blob'}).subscribe(resp => {
      this.createImageFromBlob(resp);
      this.busy = false;
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

    this.x = +x.toFixed(2);
    this.y = +y.toFixed(2);
    this.z = 0;

    this.xMove += Math.abs(event.movementX);
    this.yMove += Math.abs(event.movementY);

    if ((this.xMove + this.yMove > 20) && (!this.busy)) {
      this.postCoords(x, y, 0);
      this.xMove = 0;
      this.yMove = 0;

      this.x = +x.toFixed(2);
      this.y = +y.toFixed(2);
      this.z = 0;
    }
  }
}