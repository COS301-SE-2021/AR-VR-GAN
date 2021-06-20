import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ModelControllerClient } from '../proto/model/model_pb_service';
import { RequestDto, ResponseDto } from '../proto/model/model_pb';
import { BrowserHeaders } from 'browser-headers';

@Component({
  selector: 'app-coords',
  templateUrl: './coords.component.html',
  styleUrls: ['./coords.component.css']
})
export class CoordsComponent implements OnInit {
  public response: any;
  constructor(private http: HttpClient, private client: ModelControllerClient) { 
    this.client = new ModelControllerClient("http://localhost:3000")
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

  grpcRequest(){
    var xN: number = +x;
    var yN: number = +y;
    var zN: number = +z;
    const req = new RequestDto();
    var data = [
      xN, yN, zN
    ];
    req.addData(xN);
    req.addData(yN);
    req.addData(zN);


    this.client.handleCoords(req, "", (err, response: ResponseDto) => {
      if (err) {
        console.log("error with grpc");
      }
      console.log(response.getSum());
    });
  }





}