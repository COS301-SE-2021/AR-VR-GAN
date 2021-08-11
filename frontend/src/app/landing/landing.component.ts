import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { CookieService } from 'ngx-cookie-service';

@Component({
  selector: 'app-landing',
  templateUrl: './landing.component.html',
  styleUrls: ['./landing.component.css']
})
export class LandingComponent implements OnInit {

  constructor(private http: HttpClient, private cookieService: CookieService) { }

  ngOnInit(): void {
    if (this.cookieService.check('jwtToken')) {
      this.http.post<any>('http://localhost:3000/user/getUserByJWTToken/', {'jwtToken': this.cookieService.get('jwtToken')}).subscribe(resp => {      
      if (resp.success) {
        if (resp.user.username !== this.cookieService.get('username')) {
          window.location.replace('http://localhost:4200/login');
        }
      } else {
        window.location.replace('http://localhost:4200/login');
      }
    });
    } else {
      window.location.replace('http://localhost:4200/login');
    }
  }
}
