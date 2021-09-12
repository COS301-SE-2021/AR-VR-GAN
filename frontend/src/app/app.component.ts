import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CookieService } from 'ngx-cookie-service';
import { HOST_URL } from 'src/config/consts';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'frontend';
  loggedIn: boolean = false;

  constructor(
    private http: HttpClient, 
    private cookieService: CookieService,
    private router: Router) {
    if ((this.cookieService.check('jwtToken')) && (this.cookieService.check('username'))) {
      this.http.post<any>(HOST_URL + '/user/getUserByJWTToken/', {'jwtToken': this.cookieService.get('jwtToken')}).subscribe(resp => {      
        if (resp.success) {
          if (resp.user.username !== this.cookieService.get('username')) {
            this.loggedIn = false;
          } else {
            this.loggedIn = true;
          }
        } else {
          this.loggedIn = false;
        }
      });
    } else {
      this.loggedIn = false;
    }
  }

  ngOnInit(): void {
    if ((this.cookieService.check('jwtToken')) && (this.cookieService.check('username'))) {
      this.http.post<any>(HOST_URL + '/user/getUserByJWTToken/', {'jwtToken': this.cookieService.get('jwtToken')}).subscribe(resp => {      
        if (resp.success) {
          if (resp.user.username !== this.cookieService.get('username')) {
            this.loggedIn = false;
          } else {
            this.loggedIn = true;
          }
        } else {
          this.loggedIn = false;
        }
      });
    } else {
      this.loggedIn = false;
    }
  }

  getUsername(): string {
    if (this.cookieService.check('username')) {
      return this.cookieService.get('username');
    }

    return '';
  }

  logOut() {
    this.loggedIn = false;
    this.cookieService.deleteAll();
  }
}
