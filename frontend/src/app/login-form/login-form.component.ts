import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CookieService } from 'ngx-cookie-service';

@Component({
  selector: 'app-login-form',
  templateUrl: './login-form.component.html',
  styleUrls: ['./login-form.component.css']
})
export class LoginFormComponent implements OnInit {
  public message: string | undefined;

  constructor(private http: HttpClient, private cookieService: CookieService, private router: Router) {}

  ngOnInit(): void {}

  register(): void {
    this.router.navigate(['/register']);
  }

  loginUser(username: string, password: string): void {
    this.message = '';
    this.cookieService.deleteAll();

    this.http.post<any>('http://localhost:3000/user/login/', {'username': username, 'password': password}).subscribe(resp => {
      if (resp.success) {
        this.message = 'The user was logged in successfully.';
        this.cookieService.set('username', username);
        this.cookieService.set('jwtToken', resp.message);
        this.router.navigate(['/landing']);
      } else {
        this.message = resp.message;
      }
    });
  }
}
