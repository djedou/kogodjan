

pub trait IO {
    fn save(&self, path: &str);
    fn load(id: &str, data: &str) -> Self;
    /*

        let db = Store::new(data).unwrap();
        db.get::<T>(&id).unwrap()
    }

    fn delete(&self, id: &str, data: &str) {
        let db = Store::new(data).unwrap();
        db.delete(&id).unwrap();
    }
    */
}
