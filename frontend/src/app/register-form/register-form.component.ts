import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-register-form',
  templateUrl: './register-form.component.html',
  styleUrls: ['./register-form.component.css']
})
export class RegisterFormComponent implements OnInit {
  public message: string | undefined;

  constructor(private http: HttpClient, private router: Router) { }

  ngOnInit(): void {
  }

  login(): void {
    this.router.navigate(['/login']);
  }

  register(username: string, email: string, password: string): void {
    this.message = '';
    var valid = true;
    
    if (username.length < 1) {
      this.message += 'Please enter a username. ';
      valid = false;
    }

    if (!email.includes('@')) {
      this.message += 'Please enter a valid email address. ';
      valid = false;
    }

    if (password.length < 8) {
      this.message += 'Please enter a password that is at least 8 characters long. ';
      valid = false;
    }  

    if (valid) {
      this.http.post<any>('http://localhost:3000/user/register/', {'username': username, 'email': email, 'password': password}).subscribe(resp => {
        this.message = resp.message;
      
        if (resp.success) {
          this.router.navigate(['/login']);
        }
      });
    }
  }
}
