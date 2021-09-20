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
    link.download = "Unity_Build.zip";
    link.href = "../../../assets/Unity_Build.zip";
    link.click();
  }
}
