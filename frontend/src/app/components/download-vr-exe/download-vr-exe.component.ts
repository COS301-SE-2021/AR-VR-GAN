import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-download-vr-exe',
  templateUrl: './download-vr-exe.component.html',
  styleUrls: ['./download-vr-exe.component.css']
})
export class DownloadVrExeComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }

  downloadExe() {
    let link = document.createElement("a");
    link.download = "background.png";
    link.href = "../../../assets/background.png";
    link.click();
  }
}
