import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {

  constructor(private http: HttpClient) { }

  ngOnInit(): void {
  }


  loginUser(username: string, password: string): void {
    this.http.post('http://localhost:3000/user/login/', {'username': username, 'password': password}).subscribe(resp => {
      if (resp.success) {
        console.log('yay');
      } else {
        console.log('ah');
      }
    });
  }
}
